import numpy as np
import random
import torch
import copy
from collections import Counter
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, auc

#########################################################################################################################
################################################# PROCESS THE RAW DATAS #################################################
#########################################################################################################################

# Out-of-style!
def split_dataset_to_vl_te(graphs_dict):
    test_keys = random.sample(list(graphs_dict.keys()), round(len(graphs_dict)*0.5))

    validation_graphs = {}
    test_graphs = {}
    for k,v in graphs_dict.items():
        if k in test_keys:
            test_graphs[k] = v
        else:
            validation_graphs[k] = v
    return validation_graphs, test_graphs


def split_dataset_to_three(graphs_dict, seed, ratio=[3,1,1]):
    random.seed(seed)

    keys = list(graphs_dict.keys())
    random.shuffle(keys)
    lens = len(keys)

    denominator = sum(ratio)
    numerator_1 = ratio[0]
    numerator_2 = ratio[1] + numerator_1

    split_1 = keys[:lens*numerator_1//denominator]
    split_2 = keys[lens*numerator_1//denominator:lens*numerator_2//denominator]
    split_3 = keys[lens*numerator_2//denominator:]

    train_graphs = {k:graphs_dict[k] for k in split_1}
    validation_graphs = {k:graphs_dict[k] for k in split_2}
    test_graphs = {k:graphs_dict[k] for k in split_3}
    return train_graphs, validation_graphs, test_graphs


def refine_graphs_dict(graphs_dict, criterion = None):
    drop_list = []
    for k,v in graphs_dict.items():
        N_nodes = max(max(v["e"]["Src"]), max(v["e"]["Dst"])) + 1
        N_edges = len(v["e"])

        src = v["e"]["Src"].to_numpy()
        dst = v["e"]["Dst"].to_numpy()
        N_self = sum(np.where(src==dst, 1, 0))

        if criterion == ["NOSELF"]:
            if N_self > 0:
                drop_list.append(k)
        elif criterion == ["NEGEQALPOS"]:
            if N_nodes**2 < N_edges*2:
                drop_list.append(k)
        elif len(criterion) == 2:
            if N_nodes**2 < N_edges*2 or N_self > 0:
                drop_list.append(k)

    for k in drop_list:
        del(graphs_dict[k])
    return graphs_dict



def margin_loss(pos_score, neg_score):
    return (1 - pos_score + neg_score.view(pos_score.shape[0], -1)).clamp(min=0).mean()


##########################################################################################
############################# CHOOSE THE RANDOM SEED #####################################
##########################################################################################
def get_the_better_seed(graphs_dict, test_ratio, initial_seed):
    allos = []
    for k,v in graphs_dict.items():
        t = v['n']["OS"].tolist()
        allos += t
    res = Counter(allos)
    
    prerequesite = {}
    for k,v in res.items():
        if v < 10000:
            threshold = 0.1
            accept_range = (round((test_ratio - threshold) * v), 
                            round((test_ratio + threshold) * v))
            prerequesite[k] = {"range": accept_range, "info":{}}
        else:
            None
    
    for os in list(prerequesite.keys()):
        for k,v in graphs_dict.items():
            t = v['n']["OS"].tolist()
            if os in t:
                prerequesite[os]["info"].update({k:t.count(os)})
   
    random_seed = initial_seed
    while True:
        random.seed(random_seed)
        test_keys = random.sample(list(graphs_dict.keys()), round(len(graphs_dict)*test_ratio))
        flag_list = []
        for k,v in prerequesite.items():
            accept_range = v["range"]
            id_dict = v["info"]
            match = set(test_keys).intersection(set(id_dict))
            count_num = 0
            for mid in match:
                count_num += id_dict[mid]
            prerequesite[k].update({"num": count_num})
            
            if count_num < accept_range[0] or count_num > accept_range[1]:
                flag_list.append(False)
            else:
                flag_list.append(True)

        if False in flag_list:
            random_seed += 1
        else:
            break
    return random_seed, test_keys


def get_the_best_seed(graphs_dict, initial_seed, vl_ratio=0.2):
    while True:
        graphs_dict_copy = copy.deepcopy(graphs_dict)
        seed, target_vl_keys = get_the_better_seed(graphs_dict_copy, test_ratio=vl_ratio, initial_seed=initial_seed)
        random.seed(seed)
        vl_keys = random.sample(list(graphs_dict_copy.keys()), round(len(graphs_dict_copy)*vl_ratio))
        assert target_vl_keys == vl_keys

        [v for v in map(graphs_dict_copy.pop, vl_keys)]
        te_ratio = vl_ratio/(1-vl_ratio)
        second_seed, target_te_keys = get_the_better_seed(graphs_dict_copy, test_ratio=te_ratio, initial_seed=seed)
        random.seed(second_seed)
        te_keys = random.sample(list(graphs_dict_copy.keys()), len(target_te_keys))
        assert te_keys == target_te_keys
        
        if second_seed == seed:
            break
        else:
            initial_seed = copy.deepcopy(seed) if initial_seed != seed else (initial_seed+1)
    return second_seed,target_vl_keys,target_te_keys


def stopper(point, best, epoch, patience, lr, max_patience=30, tolerance=0.01):
    point = float(point)
    best = float(best)
    #if epoch > 80:
    if lr < 1e-7:
        if point > best+tolerance:
            best = point
            patience = 0
            save_flag = True
            stop_flag = False
        else:
            save_flag = False
            patience += 1
            if patience > max_patience:
                stop_flag = True
            else:
                stop_flag = False
    else:
        stop_flag = False
        save_flag = False
    return best, patience, stop_flag, save_flag


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True  #return the default algo
    #torch.backends.cudnn.benchmark = False     #do not find the best one
    """
    While disabling CUDA convolution benchmarking (discussed above) ensures that CUDA selects the same algorithm 
    each time an application is run, that algorithm itself may be nondeterministic, unless either 
    torch.use_deterministic_algorithms(True) or torch.backends.cudnn.deterministic = True is set.
    Outputs of the modules are non-deterministic when benchmark = True, even though deterministic = True
    When cudnn.deterministic=True. If also cudnn.benchmark=True, 
    the user should be warned that benchmarking is not possible in a deterministic implementation.
    """


#########################################################################################################################
################################################## EVALUATE THE RESULT ##################################################
#########################################################################################################################
class measure_matrix(object):
    def __init__(self):
        self.mask = []
        self.y_pred = []
        self.y_true = []
        self.graph_neutrality = []
        
    def update(self, y_pred, y_true, mask, neutrality=None):
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        self.mask.append(mask.detach().cpu())
        if neutrality != None:
            self.graph_neutrality.append(neutrality.detach().cpu())
        
    def roc_precision_recall_score(self):
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_pred = torch.sigmoid(y_pred)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            precision,  recall, _thresholds = precision_recall_curve(task_y_true, task_y_pred, pos_label=1)
            scores.append(auc(recall, precision))
        return scores
    
    def roc_auc_score(self):
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_pred = torch.sigmoid(y_pred)
        y_true = torch.cat(self.y_true, dim=0)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(roc_auc_score(task_y_true, task_y_pred))
        return scores
    
    def absolute_correct_rate(self, sample=False):
        y_pred = torch.cat(self.y_pred, dim=0)
        y_pred = torch.sigmoid(y_pred).numpy()
        y_true = torch.cat(self.y_true, dim=0).numpy()
        
        if not sample:
            score = 0
            for i in range(y_pred.shape[0]):
                t = np.where(max(y_true[i]) == y_true[i], 1, 0)
                p = np.where(max(y_pred[i]) == y_pred[i], 1, 0)
                if (t==p).all():
                    score += 1
            return score/y_pred.shape[0]
        else:
            slices = random.sample([i for i in range(y_pred.shape[0])],round(y_true.shape[0]*0.1))
            result = list(map(lambda x,y: 1 if ((x[0]-x[1])*(y[0]-y[1]))>0 else 0,y_true[slices], y_pred[slices]))
            return sum(result)/len(result)

    def regression_absolute_rate(self):
        y_pred = self.labelization(torch.round(torch.cat(self.y_pred, dim=0)))
        y_true = self.labelization(torch.cat(self.y_true, dim=0))
        y_same = [i for i, (a, b) in enumerate(zip(y_pred, y_true)) if a == b]
        r = len(y_same)/len(y_pred)
        return r

    def logistic_absolute_rate(self, threshold=None):
        y_pred = torch.cat(self.y_pred, dim=0).numpy().ravel()
        y_true = torch.cat(self.y_true, dim=0).numpy().ravel()

        if threshold == None:
            temp_y_pred = sorted(y_pred)
            m_idx = int(len(temp_y_pred)/2)
            mean = (temp_y_pred[m_idx] + temp_y_pred[m_idx+1])/2
            y_pred = np.where(y_pred >= mean, 1, 0)
        else:
            y_pred = np.where(y_pred >= 0, 1, 0)
        r = np.sum(np.where(y_pred==y_true, 1, 0)) / len(y_true)
        return r

    def average_neutrality(self):
        graph_neutrality = torch.cat(self.graph_neutrality, dim=0).numpy().ravel()
        mean_neutrality = np.mean(graph_neutrality)
        return mean_neutrality


    def BVS_MASE(self):
        y_pred = torch.cat(self.y_pred, dim=0).numpy()
        y_true = torch.cat(self.y_true, dim=0).numpy()
        y_diff = np.subtract(y_pred, y_true)
        Mean_Absolute_Error = np.mean(np.absolute(y_diff))
        Mean_Square_Error = np.mean(np.square(y_diff))
        return Mean_Absolute_Error, Mean_Square_Error

    def labelization(self, labels):
        classes=["-4","-3","-2","-1","0","1","2","3","4","5","6","7"]
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        labels = [str(int(i)) for i in labels]
        return labels

"""END HERE"""



"""

class TOSSNEGDataSet(DGLDataset):
    def __init__(self, graphs_dict, length_matrix_dict):
        self.graphs_dict = graphs_dict
        self.length_matrix_dict = length_matrix_dict
        super().__init__(name='TOSS_NEG')
    
    def process(self):
        self.graphs = []
        for k, g in self.graphs_dict.items():
            edges_data = g["e"]  
            N_node = g["n"].shape[0]
            
            edges_src = torch.from_numpy(edges_data['Src'].to_numpy())
            edges_dst = torch.from_numpy(edges_data['Dst'].to_numpy())
            
            #constract the negative graphs:
            pos_adj = sp.coo_matrix((np.ones(edges_data.shape[0]), (np.array(edges_src,dtype="int32"),
                                                       np.array(edges_dst,dtype="int32"))))
            neg_adj = 1- pos_adj.todense()
            np_src, np_dst = np.where(neg_adj != 0)

            random_index = random.sample(list(np.arange(len(np_src))), len(edges_data)//2)
            half_dst = np_dst[random_index]
            half_src = np_src[random_index]
            if len(edges_data)%2 != 0:
                neg_dst = np.hstack((half_dst,half_src,np.array(0)))
                neg_src = np.hstack((half_src,half_dst,np.array(0)))
            else:
                neg_dst = np.hstack((half_dst,half_src))
                neg_src = np.hstack((half_src,half_dst))

            assert(neg_dst.shape[0] == len(edges_data))

            ############################################
            neg_length_list = list(map(lambda x:self.length_matrix_dict[k][neg_src[x]][neg_dst[x]],[i for i in range(len(neg_dst))]))
            edge_features = torch.from_numpy(np.array(neg_length_list).astype("float32"))
            ############################################

            neg_dst = torch.tensor(neg_dst, dtype = torch.int32)
            neg_src = torch.tensor(neg_src, dtype = torch.int32)

            graph = dgl.graph((neg_src, neg_dst), num_nodes=N_node)
            graph.edata['e'] = edge_features.reshape([edge_features.shape[0],1])
            label = torch.from_numpy(np.ones((N_node, 1), dtype = "float32"))
            self.graphs.append((k, graph, label))
        
    def __getitem__(self,i):
        return self.graphs[i]
        
    def __len__(self):
        return len(self.graphs)


class TOSSPOSDataSet(DGLDataset):
    def __init__(self, graphs_dict):
        self.graphs_dict = graphs_dict
        super().__init__(name='TOSS_POS')
    
    def process(self):
        self.graphs = []
        for k, g in self.graphs_dict.items():
            nodes_data = g["n"]
            edges_data = g["e"]
            N_node = g["n"].shape[0]
        
            #node_features = torch.from_numpy(nodes_data.iloc[:,0:-1].to_numpy())
            node_features = torch.from_numpy(nodes_data.to_numpy())
            node_labels = torch.from_numpy(np.ones((N_node, 1), dtype = "float32"))

            edge_features = torch.from_numpy(edges_data['Length'].to_numpy().astype("float32"))
            edges_src = torch.from_numpy(edges_data['Src'].to_numpy())
            edges_dst = torch.from_numpy(edges_data['Dst'].to_numpy())
        
            graph = dgl.graph((edges_src, edges_dst), num_nodes=nodes_data.shape[0])
            graph.ndata['h'] = node_features
            graph.edata['e'] = edge_features.reshape([edge_features.shape[0],1])
            
            label = node_labels
            
            self.graphs.append((k, graph, label))
        
    def __getitem__(self,i):
        return self.graphs[i]
        
    def __len__(self):
        return len(self.graphs)

"""


"""
def refine_graphs_dict(graphs_dict):
    drop_list = []
    for k,v in graphs_dict.items():
        N_nodes = max(max(v["e"]["Src"]), max(v["e"]["Dst"])) + 1
        N_edges = len(v["e"])

        if N_nodes**2 < N_edges*2:
            drop_list.append(k)
    for k in drop_list:
        del(graphs_dict[k])
    return graphs_dict
"""


"""
class TOSSLPDataSet(DGLDataset):
    def __init__(self, graphs_dict, length_matrix_dict):
        self.graphs_dict = graphs_dict
        self.length_matrix_dict = length_matrix_dict
        super().__init__(name='TOSS_LP')
    
    def process(self):
        self.graphs = []
        for k, g in self.graphs_dict.items():
            nodes_data = g["n"]
            pos_edges_data = g["e"]
            N_node = g["n"].shape[0]
            
            #####################################
            ####constract the positive graphs:###
            #####################################
            pos_node_features = torch.from_numpy(nodes_data.to_numpy())
            pos_edge_features = torch.from_numpy(pos_edges_data['Length'].to_numpy().astype("float32"))
            
            pos_edges_src = torch.from_numpy(pos_edges_data['Src'].to_numpy())
            pos_edges_dst = torch.from_numpy(pos_edges_data['Dst'].to_numpy())
            
            pos_graph = dgl.graph((pos_edges_src, pos_edges_dst), num_nodes=nodes_data.shape[0])
            pos_graph.ndata['h'] = pos_node_features
            pos_graph.edata['e'] = pos_edge_features.reshape([pos_edge_features.shape[0],1])
            
            #####################################
            ####constract the negative graphs:###
            #####################################
            pos_adj = sp.coo_matrix((np.ones(pos_edges_data.shape[0]), 
                                    (np.array(pos_edges_src,dtype="int32"),
                                     np.array(pos_edges_dst,dtype="int32"))))
            neg_adj = 1- pos_adj.todense()
            np_neg_src, np_neg_dst = np.where(neg_adj != 0)

            random_index = random.sample(list(np.arange(len(np_neg_src))), len(pos_edges_data)//2)
            half_neg_dst = np_neg_dst[random_index]
            half_neg_src = np_neg_src[random_index]
            
            if len(pos_edges_data)%2 != 0:
                neg_dst = np.hstack((half_neg_dst,half_neg_src,np.array(0)))
                neg_src = np.hstack((half_neg_src,half_neg_dst,np.array(0)))
            else:
                neg_dst = np.hstack((half_neg_dst,half_neg_src))
                neg_src = np.hstack((half_neg_src,half_neg_dst))

            assert(neg_dst.shape[0] == len(pos_edges_data))

            neg_length_list = list(map(lambda x:self.length_matrix_dict[k][neg_src[x]][neg_dst[x]],[i for i in range(len(neg_dst))]))
            neg_edge_features = torch.from_numpy(np.array(neg_length_list).astype("float32"))

            neg_dst = torch.tensor(neg_dst, dtype = torch.int32)
            neg_src = torch.tensor(neg_src, dtype = torch.int32)

            neg_graph = dgl.graph((neg_src, neg_dst), num_nodes=nodes_data.shape[0])
            neg_graph.edata['e'] = neg_edge_features.reshape([neg_edge_features.shape[0],1])
            ########################################FINISH##################################
            
            self.graphs.append((k, pos_graph, neg_graph))
        
    def __getitem__(self,i):
        return self.graphs[i]
        
    def __len__(self):
        return len(self.graphs)
"""