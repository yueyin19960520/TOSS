import torch
import dgl
import scipy.sparse as sp
import numpy as np
from collections import Counter
from dgl.data import DGLDataset

def encode_onehot(labels):
    classes=["-4","-3","-2","-1","0","1","2","3","4","5","6","7"]
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels = [str(int(i)) for i in labels]
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

##############################################################################################
################################ Node Classfication ##########################################
##############################################################################################
class TOSS_DGL_NC_LEF_DataSet(DGLDataset):  
    # LEF:less edge features
    # only consider the bond length as the edge features.
    def __init__(self, graphs_dict):
        self.graphs_dict = graphs_dict
        super().__init__(name='TOSS_NC_LEF')
    
    def process(self):
        self.graphs = []
        for k, g in self.graphs_dict.items():
            nodes_data = g["n"]
            edges_data = g["e"]
        
            nodes_features = torch.from_numpy(nodes_data.iloc[:,0:-1].to_numpy())
            nodes_labels = torch.from_numpy(np.array(encode_onehot(nodes_data["OS"]), dtype = "float32"))

            edges_features = torch.from_numpy(edges_data['Length'].to_numpy().astype("float32"))
            edges_src = torch.from_numpy(edges_data['Src'].to_numpy())
            edges_dst = torch.from_numpy(edges_data['Dst'].to_numpy())
        
            graph = dgl.graph((edges_src, edges_dst), num_nodes=nodes_data.shape[0])
            graph.ndata['h'] = nodes_features
            graph.edata['e'] = edges_features.reshape([edges_features.shape[0],1])
            
            label = nodes_labels
            
            self.graphs.append((k, graph, label))
        
    def __getitem__(self,i):
        return self.graphs[i]
        
    def __len__(self):
        return len(self.graphs)


class TOSS_DGL_NC_MEF_DataSet(DGLDataset):  
    # MEF: more edge features
    # New dataset with more bond datas
    # Consider different and transfromed bond length as the edge features.
    def __init__(self, graphs_dict):
        self.graphs_dict = graphs_dict
        super().__init__(name='TOSS_NC_MEF')
    
    def process(self):
        self.graphs = []
        for k, g in self.graphs_dict.items():
            nodes_data = g["n"]
            edges_data = g["e"]
        
            nodes_features = torch.from_numpy(nodes_data.iloc[:,0:-1].to_numpy())
            nodes_labels = torch.from_numpy(np.array(encode_onehot(nodes_data["OS"]), dtype = "float32"))

            edges_src = torch.from_numpy(edges_data['Src'].to_numpy())
            edges_dst = torch.from_numpy(edges_data['Dst'].to_numpy())
            bond_idx = np.arange(len(edges_data))

            EN = list(nodes_data["EN"])
            R1 = list(nodes_data["R1"])
            R2 = list(nodes_data["R2"])
            R3 = list(nodes_data["R3"])
            IP1 = list(nodes_data["IP1"])
            IP2 = list(nodes_data["IP2"])
            IP3 = list(nodes_data["IP3"])

            length_list = list(map(lambda x:edges_data['Length'][x], bond_idx))
            delta_EN_list = list(map(lambda x:abs(EN[edges_src[x]]-EN[edges_dst[x]]), bond_idx))
            sum_R1_list = list(map(lambda x:abs(R1[edges_src[x]]+R1[edges_dst[x]])*0.01,bond_idx))
            sum_R2_list = list(map(lambda x:abs(R2[edges_src[x]]+R2[edges_dst[x]])*0.01,bond_idx))
            sum_R3_list = list(map(lambda x:abs(R3[edges_src[x]]+R3[edges_dst[x]])*0.01,bond_idx))
            delta_IP1_list = list(map(lambda x:abs(IP1[edges_src[x]]-IP1[edges_dst[x]])*0.001,bond_idx))
            delta_IP2_list = list(map(lambda x:abs(IP2[edges_src[x]]-IP2[edges_dst[x]])*0.001,bond_idx))
            delta_IP3_list = list(map(lambda x:abs(IP3[edges_src[x]]-IP3[edges_dst[x]])*0.001,bond_idx))
            ratio1_list = list(map(lambda x:(edges_data['Length'][x]/abs(0.01*(R1[edges_src[x]]+R1[edges_dst[x]]))),bond_idx))
            ratio2_list = list(map(lambda x:(edges_data['Length'][x]/abs(0.01*(R2[edges_src[x]]+R2[edges_dst[x]]))),bond_idx))
            ratio3_list = list(map(lambda x:(edges_data['Length'][x]/abs(0.01*(R3[edges_src[x]]+R3[edges_dst[x]]))),bond_idx))
            one_over_l_list = list(map(lambda x:1/(edges_data['Length'][x]+1),bond_idx))
            square_l_list = list(map(lambda x:(edges_data['Length'][x])**2,bond_idx))

            edges_features = torch.from_numpy(np.hstack((np.array(length_list).reshape(len(length_list),1),
                                                         np.array(one_over_l_list).reshape(len(one_over_l_list),1),
                                                         np.array(square_l_list).reshape(len(square_l_list),1),
                                                         np.array(ratio1_list).reshape(len(ratio1_list),1),
                                                         np.array(ratio2_list).reshape(len(ratio2_list),1),
                                                         np.array(ratio3_list).reshape(len(ratio3_list),1),
                                                         np.array(delta_EN_list).reshape(len(delta_EN_list),1),
                                                         np.array(delta_IP1_list).reshape(len(delta_IP1_list),1),
                                                         np.array(delta_IP2_list).reshape(len(delta_IP2_list),1),
                                                         np.array(delta_IP3_list).reshape(len(delta_IP3_list),1),
                                                         np.array(sum_R1_list).reshape(len(sum_R1_list),1),
                                                         np.array(sum_R2_list).reshape(len(sum_R2_list),1),
                                                         np.array(sum_R3_list).reshape(len(sum_R3_list),1))).astype("float32"))

            graph = dgl.graph((edges_src, edges_dst), num_nodes=nodes_data.shape[0])
            graph.ndata['h'] = nodes_features
            graph.edata['e'] = edges_features
            
            label = nodes_labels
            
            self.graphs.append((k, graph, label))
        
    def __getitem__(self,i):
        return self.graphs[i]
        
    def __len__(self):
        return len(self.graphs)


##############################################################################################
################################## Link Prediction ###########################################
##############################################################################################
class TOSS_DGL_LP_NP_DataSet(DGLDataset):  
    # NP : negatibe and positibe graphs 
    # construct two graphs containing negative and positive graphs
    def __init__(self, graphs_dict, length_matrix_dict):
        self.graphs_dict = graphs_dict
        self.length_matrix_dict = length_matrix_dict
        super().__init__(name='TOSS_LP_NP')
    
    def process(self):
        self.graphs = []
        for k, g in self.graphs_dict.items():
            nodes_data = g["n"]
            pos_edges_data = g["e"]
            N_node = g["n"].shape[0]

            #positive nodes determination:
            pos_node_features = torch.from_numpy(nodes_data.to_numpy())
            pos_edge_features = torch.from_numpy(pos_edges_data['Length'].to_numpy().astype("float32"))

            pos_src = pos_edges_data['Src'].to_numpy()
            pos_dst = pos_edges_data['Dst'].to_numpy()

            num_pos_edge = pos_src.shape[0]
            assert num_pos_edge % 2 == 0

            pos_adj = sp.coo_matrix((np.ones(num_pos_edge), 
                                    (np.array(pos_src,dtype="int32"),
                                     np.array(pos_dst,dtype="int32"))))

            neg_adj = 1 - pos_adj.todense() - np.eye(pos_adj.shape[0])
            neg_src, neg_dst = np.where(neg_adj != 0)
            num_neg_edge = len(neg_src)

            if num_pos_edge < num_neg_edge:
                uni_neg_src = neg_src[(neg_dst > neg_src)]
                uni_neg_dst = neg_dst[(neg_dst > neg_src)]
                sample_index = random.sample(list(np.arange(len(uni_neg_src))), num_pos_edge//2)
                random_neg_src = uni_neg_src[sample_index]
                random_neg_dst = uni_neg_dst[sample_index]

                neg_src = np.hstack((random_neg_src, random_neg_dst))
                neg_dst = np.hstack((random_neg_dst, random_neg_src))
            else:
                uni_pos_src = pos_src[(pos_dst > pos_src)]
                uni_pos_dst = pos_dst[(pos_dst > pos_src)]
                sample_index = random.sample(list(np.arange(len(uni_pos_src))), num_neg_edge//2)
                random_pos_src = uni_pos_src[sample_index]
                random_pos_dst = uni_pos_dst[sample_index]

                pos_src = np.hstack((random_pos_src, random_pos_dst))
                pos_dst = np.hstack((random_pos_dst, random_pos_src))

            assert len(pos_src) == len(neg_dst) == len(pos_dst) == len(neg_src)

            pos_length_list = list(map(lambda x:self.length_matrix_dict[k][pos_src[x]][pos_dst[x]],[i for i in range(len(pos_dst))]))
            pos_edge_features = torch.from_numpy(np.array(pos_length_list).astype("float32"))

            neg_length_list = list(map(lambda x:self.length_matrix_dict[k][neg_src[x]][neg_dst[x]],[i for i in range(len(neg_dst))]))
            neg_edge_features = torch.from_numpy(np.array(neg_length_list).astype("float32"))

            pos_src = torch.from_numpy(pos_src)
            pos_dst = torch.from_numpy(pos_dst)
            neg_src = torch.from_numpy(neg_src)
            neg_dst = torch.from_numpy(neg_dst) #neg_src = torch.tensor(neg_src, dtype = torch.int32)

            ####constract the positive graphs:###
            pos_graph = dgl.graph((pos_src, pos_dst), num_nodes=nodes_data.shape[0])
            pos_graph.ndata['h'] = pos_node_features
            pos_graph.edata['e'] = pos_edge_features.reshape([pos_edge_features.shape[0],1])
            ####constract the negative graphs:###
            neg_graph = dgl.graph((neg_src, neg_dst), num_nodes=nodes_data.shape[0])
            neg_graph.edata['e'] = neg_edge_features.reshape([neg_edge_features.shape[0],1])
            ########################################FINISH##################################
            
            self.graphs.append((k, pos_graph, neg_graph))
        
    def __getitem__(self,i):
        return self.graphs[i]
        
    def __len__(self):
        return len(self.graphs)



class TOSS_DGL_LP_SG_DataSet(DGLDataset):
    # SG: single graph
    # construct a single graph contains both bonds and non-bonds
    def __init__(self, graphs_dict, length_matrix_dict):
        self.graphs_dict = graphs_dict
        self.length_matrix_dict = length_matrix_dict
        super().__init__(name='TOSS_LP_SG')
    
    def process(self):
        self.graphs = []
        for k,g in self.graphs_dict.items():
            nodes_data = g["n"]
            pos_edges_data = g["e"]
            N_node = nodes_data.shape[0]
            node_features = torch.from_numpy(nodes_data.to_numpy())

            #positive nodes determination:
            pos_edge_features = torch.from_numpy(pos_edges_data['Length'].to_numpy().astype("float32"))
            pos_src = pos_edges_data['Src'].to_numpy()
            pos_dst = pos_edges_data['Dst'].to_numpy()

            num_pos_edge = pos_src.shape[0]

            pos_adj = sp.coo_matrix((np.ones(num_pos_edge), 
                                    (np.array(pos_src,dtype="int32"),
                                     np.array(pos_dst,dtype="int32"))))
            temp_pos_adj = np.array(pos_adj.todense(), dtype="int32")

            try:
                local_adj = np.where(self.length_matrix_dict[k] <= 5, 1, 0)
                dense_temp_neg_adj = np.where(local_adj - temp_pos_adj == 1, 1, 0)
                coo_temp_neg_adj = sp.coo_matrix(dense_temp_neg_adj)


                pair = list(filter(None, 
                                   list(map(lambda x:x if x[1] > x[0] else None, 
                                            list(map(lambda x,y: (x,y), 
                                                     coo_temp_neg_adj.row, 
                                                     coo_temp_neg_adj.col))))))
                try:
                    selected_pair = random.sample(pair, round((num_pos_edge - N_node)/2))
                except:
                    selected_pair = pair
                neg_pair_without_self_loop = np.array(selected_pair + list(map(lambda x:(x[1],x[0]), 
                                                                               selected_pair)))

                neg_adj = sp.coo_matrix((np.ones(len(neg_pair_without_self_loop)), 
                                        (np.array(neg_pair_without_self_loop[:,0],dtype="int32"), 
                                         np.array(neg_pair_without_self_loop[:,1],dtype="int32")))).todense() + np.eye(N_node)
                neg_adj = sp.coo_matrix(neg_adj)
                neg_src = neg_adj.row
                neg_dst = neg_adj.col

                edge_labels = torch.from_numpy(np.vstack((np.hstack((np.ones((len(pos_src),1)),
                                                                     np.zeros((len(pos_dst),1)))),
                                                          np.hstack((np.zeros((len(neg_src),1)),
                                                                     np.ones((len(neg_dst),1)))))))

                pos_src = torch.from_numpy(pos_src)
                pos_dst = torch.from_numpy(pos_dst)
                neg_src = torch.from_numpy(neg_src)
                neg_dst = torch.from_numpy(neg_dst)

                src = torch.concat((pos_src, neg_src))
                dst = torch.concat((pos_dst, neg_dst))

                EN = list(nodes_data["EN"])
                R1 = list(nodes_data["R1"])
                R2 = list(nodes_data["R2"])
                R3 = list(nodes_data["R3"])
                IP1 = list(nodes_data["IP1"])
                IP2 = list(nodes_data["IP2"])
                IP3 = list(nodes_data["IP3"])

                length_list = list(map(lambda x:self.length_matrix_dict[k][src[x]][dst[x]],[i for i in range(len(dst))]))
                delta_EN_list = list(map(lambda x:abs(EN[src[x]]-EN[dst[x]]),[i for i in range(len(dst))]))
                sum_R1_list = list(map(lambda x:abs(R1[src[x]]+R1[dst[x]])*0.01,[i for i in range(len(dst))]))
                sum_R2_list = list(map(lambda x:abs(R2[src[x]]+R2[dst[x]])*0.01,[i for i in range(len(dst))]))
                sum_R3_list = list(map(lambda x:abs(R3[src[x]]+R3[dst[x]])*0.01,[i for i in range(len(dst))]))
                delta_IP1_list = list(map(lambda x:abs(IP1[src[x]]-IP1[dst[x]]),[i for i in range(len(dst))]))
                delta_IP2_list = list(map(lambda x:abs(IP2[src[x]]-IP2[dst[x]]),[i for i in range(len(dst))]))
                delta_IP3_list = list(map(lambda x:abs(IP3[src[x]]-IP3[dst[x]]),[i for i in range(len(dst))]))
                ratio1_list = list(map(lambda x:(self.length_matrix_dict[k][src[x]][dst[x]]/abs(0.01*(R1[src[x]]+R1[dst[x]]))),[i for i in range(len(dst))]))


                edge_features = torch.from_numpy(np.hstack((np.array(length_list).reshape(len(length_list),1),
                                                            np.array(ratio1_list).reshape(len(ratio1_list),1),
                                                            np.array(delta_EN_list).reshape(len(delta_EN_list),1),
                                                            np.array(delta_IP1_list).reshape(len(delta_IP1_list),1),
                                                            np.array(delta_IP2_list).reshape(len(delta_IP2_list),1),
                                                            np.array(delta_IP3_list).reshape(len(delta_IP3_list),1),
                                                            np.array(sum_R1_list).reshape(len(sum_R1_list),1),
                                                            np.array(sum_R2_list).reshape(len(sum_R2_list),1),
                                                            np.array(sum_R3_list).reshape(len(sum_R3_list),1))).astype("float32"))
                #edge_labels = torch.from_numpy((np.vstack((np.ones((pos_edge_features.shape[0],1)),np.zeros((neg_edge_features.shape[0],1))))))

                graph = dgl.graph((src, dst), num_nodes=nodes_data.shape[0])
                graph.ndata['h'] = node_features
                graph.edata['e'] = edge_features.reshape([edge_features.shape[0],-1])
                
                self.graphs.append((k, graph, edge_labels))
            except:
                None
            
    def __getitem__(self,i):
        return self.graphs[i]
        
    def __len__(self):
        return len(self.graphs)


class TOSS_DGL_LP_FN_DataSet(DGLDataset):
    # FN : fake node
    # considering the edge as a new kind of node to build a hetero graph
    def __init__(self, graphs_dict, length_matrix_dict):
        self.graphs_dict = graphs_dict
        self.length_matrix_dict = length_matrix_dict
        super().__init__(name='TOSS_LP_FN')
    
    def process(self):
        self.graphs = []
        for k,g in self.graphs_dict.items():
            atoms_data = g["n"]
            pos_bonds_data = g["e"]
            num_atoms = atoms_data.shape[0]
            atoms_features = torch.from_numpy(atoms_data.to_numpy())            

            #positive nodes determination:
            pos_src = pos_bonds_data['Src'].to_numpy().astype("int32")
            pos_dst = pos_bonds_data['Dst'].to_numpy().astype("int32")
            num_pos_bonds = pos_src.shape[0]

            pos_adj = sp.coo_matrix((np.ones(num_pos_bonds), (np.array(pos_src), np.array(pos_dst))), shape=(num_atoms, num_atoms))
            dense_pos_adj = np.array(pos_adj.todense(), dtype="int32")

            #local_adj = np.where(self.length_matrix_dict[k] <= 5, 1, 0)
            local = np.vstack(list(map(lambda x:np.where(x<1.5*(sorted(x)[1]),1,0),self.length_matrix_dict[k])))
            trans_local = local.transpose()
            diff_local = trans_local - local
            mask = np.where(diff_local==1, 1, 0)
            local_adj = local + mask   #symmetry            

            neg_adj = sp.coo_matrix(np.where(local_adj - dense_pos_adj == 1, 1, 0).astype("float64"))

            neg_src = neg_adj.row
            neg_dst = neg_adj.col

            src = np.hstack((pos_src,neg_src))
            dst = np.hstack((pos_dst,neg_dst))
            num_bonds = len(src)

            bond_idx = np.arange(num_bonds, dtype="int32")
            src_bond_dst = np.vstack((src, bond_idx, dst))
            atoms_bonds = np.vstack((np.hstack((src_bond_dst[0], src_bond_dst[-1])),np.hstack((bond_idx, bond_idx))))
    
            graph_data = {('atoms', 'interacts', 'bonds'): (torch.from_numpy(atoms_bonds[0]), 
                                                            torch.from_numpy(atoms_bonds[1])),
                          ('bonds', 'interacts', 'atoms'): (torch.from_numpy(atoms_bonds[1]),
                                                            torch.from_numpy(atoms_bonds[0])),
                          ('bonds', 'interacts', 'bonds'): (torch.from_numpy(bond_idx),
                                                            torch.from_numpy(bond_idx)),
                          ('atoms', 'interacts', 'atoms'): (torch.from_numpy(np.arange(num_atoms, dtype="int32")),
                                                            torch.from_numpy(np.arange(num_atoms, dtype="int32")))}
            EN = list(atoms_data["EN"])
            R1 = list(atoms_data["R1"])
            R2 = list(atoms_data["R2"])
            R3 = list(atoms_data["R3"])
            IP1 = list(atoms_data["IP1"])
            IP2 = list(atoms_data["IP2"])
            IP3 = list(atoms_data["IP3"])

            length_list = list(map(lambda x:self.length_matrix_dict[k][src_bond_dst[0,x]][src_bond_dst[2,x]], bond_idx))
            delta_EN_list = list(map(lambda x:abs(EN[src_bond_dst[0,x]]-EN[src_bond_dst[2,x]]), bond_idx))
            sum_R1_list = list(map(lambda x:abs(R1[src_bond_dst[0,x]]+R1[src_bond_dst[2,x]])*0.01,bond_idx))
            sum_R2_list = list(map(lambda x:abs(R2[src_bond_dst[0,x]]+R2[src_bond_dst[2,x]])*0.01,bond_idx))
            sum_R3_list = list(map(lambda x:abs(R3[src_bond_dst[0,x]]+R3[src_bond_dst[2,x]])*0.01,bond_idx))
            delta_IP1_list = list(map(lambda x:abs(IP1[src_bond_dst[0,x]]-IP1[src_bond_dst[2,x]])*0.001,bond_idx))
            delta_IP2_list = list(map(lambda x:abs(IP2[src_bond_dst[0,x]]-IP2[src_bond_dst[2,x]])*0.001,bond_idx))
            delta_IP3_list = list(map(lambda x:abs(IP3[src_bond_dst[0,x]]-IP3[src_bond_dst[2,x]])*0.001,bond_idx))
            ratio1_list = list(map(lambda x:(self.length_matrix_dict[k][src_bond_dst[0,x]][src_bond_dst[2,x]]/abs(0.01*(R1[src_bond_dst[0,x]]+R1[src_bond_dst[2,x]]))),bond_idx))
            ratio2_list = list(map(lambda x:(self.length_matrix_dict[k][src_bond_dst[0,x]][src_bond_dst[2,x]]/abs(0.01*(R2[src_bond_dst[0,x]]+R2[src_bond_dst[2,x]]))),bond_idx))
            ratio3_list = list(map(lambda x:(self.length_matrix_dict[k][src_bond_dst[0,x]][src_bond_dst[2,x]]/abs(0.01*(R3[src_bond_dst[0,x]]+R3[src_bond_dst[2,x]]))),bond_idx))
            one_over_l_list = list(map(lambda x:1/(self.length_matrix_dict[k][src_bond_dst[0,x]][src_bond_dst[2,x]]+1),bond_idx))
            square_l_list = list(map(lambda x:(self.length_matrix_dict[k][src_bond_dst[0,x]][src_bond_dst[2,x]])**2,bond_idx))

            bonds_features = torch.from_numpy(np.hstack((np.array(length_list).reshape(len(length_list),1),
                                                         np.array(one_over_l_list).reshape(len(one_over_l_list),1),
                                                         np.array(square_l_list).reshape(len(square_l_list),1),
                                                         np.array(ratio1_list).reshape(len(ratio1_list),1),
                                                         np.array(ratio2_list).reshape(len(ratio2_list),1),
                                                         np.array(ratio3_list).reshape(len(ratio3_list),1),
                                                         np.array(delta_EN_list).reshape(len(delta_EN_list),1),
                                                         np.array(delta_IP1_list).reshape(len(delta_IP1_list),1),
                                                         np.array(delta_IP2_list).reshape(len(delta_IP2_list),1),
                                                         np.array(delta_IP3_list).reshape(len(delta_IP3_list),1),
                                                         np.array(sum_R1_list).reshape(len(sum_R1_list),1),
                                                         np.array(sum_R2_list).reshape(len(sum_R2_list),1),
                                                         np.array(sum_R3_list).reshape(len(sum_R3_list),1))).astype("float32"))
    
            bonds_labels = torch.from_numpy(np.vstack((np.hstack((np.ones((len(pos_src),1)),
                                                                  np.zeros((len(pos_dst),1)))),
                                                       np.hstack((np.zeros((len(neg_src),1)),
                                                                  np.ones((len(neg_dst),1)))))))
            graph = dgl.heterograph(graph_data)
            graph.nodes["atoms"].data['h'] = atoms_features
            graph.nodes["bonds"].data["h"] = bonds_features
                
            self.graphs.append((k, graph, bonds_labels))
            
    def __getitem__(self,i):
        return self.graphs[i]
        
    def __len__(self):
        return len(self.graphs)



##########################################################################################
################################ Collate Functions #######################################
##########################################################################################
def DGL_NC_collate(data):
    if len(data[0]) == 3:
        mids, graphs, labels = map(list, zip(*data))
        masks = None
    else:
        mids, graphs, labels, masks = map(list, zip(*data))
    
    new_graphs = []
    for graph in graphs:
        g = dgl.add_self_loop(graph)
        new_graphs.append(g)

    bg = dgl.batch(new_graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)

    labels = torch.vstack(labels)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)
    return mids, bg, labels, masks


def DGL_LP_NP_collate(data):
    mids, pos_graphs, neg_graphs = map(list, zip(*data))
    masks = None
    
    new_pos_graphs = []
    new_neg_graphs = []
    for pos_graph, neg_graph in zip(pos_graphs, neg_graphs):
        pos_g = dgl.add_self_loop(pos_graph)
        neg_g = dgl.add_self_loop(neg_graph)
        new_pos_graphs.append(pos_g)
        new_neg_graphs.append(neg_g)

    pos_bg = dgl.batch(new_pos_graphs)
    pos_bg.set_n_initializer(dgl.init.zero_initializer)
    pos_bg.set_e_initializer(dgl.init.zero_initializer)
    
    neg_bg = dgl.batch(new_neg_graphs)
    neg_bg.set_n_initializer(dgl.init.zero_initializer)
    neg_bg.set_e_initializer(dgl.init.zero_initializer)
    
    return mids, pos_bg, neg_bg


def DGL_LP_SG_collate(data):
    mids, graphs, labels = map(list, zip(*data))
    masks = None
    
    new_graphs = []
    for graph in graphs:
        #g = dgl.add_self_loop(graph)
        new_graphs.append(graph)
    
    bg = dgl.batch(new_graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    
    labels = torch.vstack(labels)
    
    return mids, bg, labels


def DGL_LP_FN_collate(data):
    mids, graphs, labels = map(list, zip(*data))
    masks = None
    
    new_graphs = []
    for graph in graphs:
        #g = dgl.add_self_loop(graph)
        new_graphs.append(graph)
    
    bg = dgl.batch(new_graphs)
    bg.set_n_initializer(dgl.init.zero_initializer,ntype="bonds")
    bg.set_n_initializer(dgl.init.zero_initializer,ntype="atoms")
    bg.set_e_initializer(dgl.init.zero_initializer,etype=('atoms', 'interacts', 'bonds'))
    bg.set_e_initializer(dgl.init.zero_initializer,etype=('bonds', 'interacts', 'atoms'))
    bg.set_e_initializer(dgl.init.zero_initializer,etype=('bonds', 'interacts', 'bonds'))
    bg.set_e_initializer(dgl.init.zero_initializer,etype=('atoms', 'interacts', 'atoms'))
    
    labels = torch.vstack(labels)
    
    return mids, bg, labels
"""END HERE"""