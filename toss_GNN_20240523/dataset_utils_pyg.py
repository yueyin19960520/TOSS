import numpy as np
import torch
import pickle
import scipy.sparse as sp
from torch_geometric.data import Dataset as PYGDataset
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from data_utils import refine_graphs_dict


def encode_onehot(labels):
    classes=["-4","-3","-2","-1","0","1","2","3","4","5","6","7"]
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels = [str(int(i)) for i in labels]
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

########################################################################################################################################
##################################################### Node Classfication ###############################################################
########################################################################################################################################
class TOSS_PYG_NC_MEF_DataSet(PYGDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(TOSS_PYG_NC_MEF_DataSet, self).__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["graphs_dict.pkl"]

    @property
    def processed_file_names(self):
        return ["NN_NC_PYG_dataset.pt"]

    def download(self):
        pass

    def process(self):
        # load data
        with open(self.raw_paths[0], 'rb') as f:
            graphs_dict = pickle.load(f)
        
        data_list = []
        for k, g in graphs_dict.items():
            nodes_data = g["n"]
            edges_data = g["e"]

            nodes_features = torch.from_numpy(nodes_data.iloc[:,0:-1].to_numpy())
            nodes_labels = torch.from_numpy(np.array(encode_onehot(nodes_data["OS"]), dtype = "float32"))

            edges_src = edges_data['Src']
            edges_dst = edges_data['Dst']
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
            # In PyG, edges are defined in pairs
            edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
            
            data = Data(x=nodes_features, edge_index=edge_index, edge_attr=edges_features, y=nodes_labels)
            data_list.append(data)  
            # self.data, self.slices = self.collate(data_list)
        
        # collate all your processed data and save it into a single file
        torch.save(data_list, self.processed_paths[0])
 
    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]


#######################################################################################################################################
####################################################### Bond Valence Regression #######################################################
#######################################################################################################################################
class TOSS_PYG_BVR_MEF_DataSet(PYGDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(TOSS_PYG_BVR_MEF_DataSet, self).__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["graphs_dict.pkl"]

    @property
    def processed_file_names(self):
        return ["NN_BVR_PYG_dataset.pt"]

    def download(self):
        pass

    def process(self):
        # load data
        with open(self.raw_paths[0], 'rb') as f:
            graphs_dict = pickle.load(f)
        
        data_list = []
        for k, g in graphs_dict.items():
            nodes_data = g["n"]
            edges_data = g["e"]

            nodes_features = torch.from_numpy(nodes_data.iloc[:,0:-1].to_numpy())
            nodes_labels = torch.from_numpy(np.array(nodes_data["OS"], dtype = "float32"))
            #nodes_labels = torch.from_numpy(np.array(encode_onehot(nodes_data["OS"]), dtype = "float32"))

            edges_src = edges_data['Src']
            edges_dst = edges_data['Dst']
            edge_weights = torch.from_numpy(np.array([1. if i>j else -1. for i,j in zip(edges_src, edges_dst)], dtype = "float32"))
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
            # In PyG, edges are defined in pairs
            edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
            
            data = Data(x=nodes_features, edge_index=edge_index, edge_attr=edges_features, y=nodes_labels, edge_weights=edge_weights)
            data_list.append(data)  
            # self.data, self.slices = self.collate(data_list)
        
        # collate all your processed data and save it into a single file
        torch.save(data_list, self.processed_paths[0])
 
    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]


#######################################################################################################################################
########################################################### Link Prediction ###########################################################
#######################################################################################################################################

class TOSS_PYG_LP_FN_DataSet(PYGDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(TOSS_PYG_LP_FN_DataSet, self).__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["graphs_dict.pkl", "length_matrix_dict.pkl"]

    @property
    def processed_file_names(self):
        return ["NN_LP_PYG_dataset.pt"]

    def download(self):
        pass

    def process(self):
        # load data
        with open(self.raw_paths[0], 'rb') as f1:
            graphs_dict = pickle.load(f1)
            graphs_dict = refine_graphs_dict(graphs_dict, criterion = ["NOSELF"])
            None_list = list(map(lambda x:graphs_dict[x]["n"].drop(["CN","SEN","OS"], axis=1, inplace=True), list(graphs_dict.keys())))

        with open(self.raw_paths[1], 'rb') as f2:
            length_matrix_dict = pickle.load(f2)
        
        data_list = []
        for k, g in graphs_dict.items():
            atoms_data = g["n"]
            pos_bonds_data = g["e"]
            num_atoms = atoms_data.shape[0]
            atoms_features = torch.from_numpy(atoms_data.to_numpy())   

            #positive nodes determination:
            pos_src = pos_bonds_data['Src'].to_numpy().astype("int64")
            pos_dst = pos_bonds_data['Dst'].to_numpy().astype("int64")
            num_pos_bonds = pos_src.shape[0]
            
            pos_adj = sp.coo_matrix((np.ones(num_pos_bonds), (np.array(pos_src), np.array(pos_dst))), shape=(num_atoms, num_atoms))
            dense_pos_adj = np.array(pos_adj.todense(), dtype="int64")  
            
            #local_adj = np.where(self.length_matrix_dict[k] <= 5, 1, 0)
            local = np.vstack(list(map(lambda x:np.where(x<1.5*(sorted(x)[1]),1,0),length_matrix_dict[k])))
            trans_local = local.transpose()
            diff_local = trans_local - local
            mask = np.where(diff_local==1, 1, 0)
            local_adj = local + mask   #symmetry            

            neg_adj = sp.coo_matrix(np.where(local_adj - dense_pos_adj == 1, 1, 0).astype("float64"))

            neg_src = neg_adj.row.astype("int64")
            neg_dst = neg_adj.col.astype("int64")

            src = np.hstack((pos_src,neg_src))
            dst = np.hstack((pos_dst,neg_dst))
            num_bonds = len(src)

            bond_idx = np.arange(num_bonds, dtype="int64")
            src_bond_dst = np.vstack((src, bond_idx, dst))
            atoms_bonds = np.vstack((np.hstack((src_bond_dst[0], src_bond_dst[-1])),np.hstack((bond_idx, bond_idx))))

            EN = list(atoms_data["EN"])
            R1 = list(atoms_data["R1"])
            R2 = list(atoms_data["R2"])
            R3 = list(atoms_data["R3"])
            IP1 = list(atoms_data["IP1"])
            IP2 = list(atoms_data["IP2"])
            IP3 = list(atoms_data["IP3"])

            length_list = list(map(lambda x:length_matrix_dict[k][src_bond_dst[0,x]][src_bond_dst[2,x]], bond_idx))
            delta_EN_list = list(map(lambda x:abs(EN[src_bond_dst[0,x]]-EN[src_bond_dst[2,x]]), bond_idx))
            sum_R1_list = list(map(lambda x:abs(R1[src_bond_dst[0,x]]+R1[src_bond_dst[2,x]])*0.01,bond_idx))
            sum_R2_list = list(map(lambda x:abs(R2[src_bond_dst[0,x]]+R2[src_bond_dst[2,x]])*0.01,bond_idx))
            sum_R3_list = list(map(lambda x:abs(R3[src_bond_dst[0,x]]+R3[src_bond_dst[2,x]])*0.01,bond_idx))
            delta_IP1_list = list(map(lambda x:abs(IP1[src_bond_dst[0,x]]-IP1[src_bond_dst[2,x]])*0.001,bond_idx))
            delta_IP2_list = list(map(lambda x:abs(IP2[src_bond_dst[0,x]]-IP2[src_bond_dst[2,x]])*0.001,bond_idx))
            delta_IP3_list = list(map(lambda x:abs(IP3[src_bond_dst[0,x]]-IP3[src_bond_dst[2,x]])*0.001,bond_idx))
            ratio1_list = list(map(lambda x:(length_matrix_dict[k][src_bond_dst[0,x]][src_bond_dst[2,x]]/abs(0.01*(R1[src_bond_dst[0,x]]+R1[src_bond_dst[2,x]]))),bond_idx))
            ratio2_list = list(map(lambda x:(length_matrix_dict[k][src_bond_dst[0,x]][src_bond_dst[2,x]]/abs(0.01*(R2[src_bond_dst[0,x]]+R2[src_bond_dst[2,x]]))),bond_idx))
            ratio3_list = list(map(lambda x:(length_matrix_dict[k][src_bond_dst[0,x]][src_bond_dst[2,x]]/abs(0.01*(R3[src_bond_dst[0,x]]+R3[src_bond_dst[2,x]]))),bond_idx))
            one_over_l_list = list(map(lambda x:1/(length_matrix_dict[k][src_bond_dst[0,x]][src_bond_dst[2,x]]+1),bond_idx))
            square_l_list = list(map(lambda x:(length_matrix_dict[k][src_bond_dst[0,x]][src_bond_dst[2,x]])**2,bond_idx))

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
            ####################################
            data = HeteroData()
            data['atoms'].x = atoms_features
            data['bonds'].x = bonds_features
            data['atoms', 'interacts', 'bonds'].edge_index = torch.vstack([torch.from_numpy(atoms_bonds[0]), 
                                                                           torch.from_numpy(atoms_bonds[1])])
            data['bonds', 'interacts', 'atoms'].edge_index = torch.vstack([torch.from_numpy(atoms_bonds[1]), 
                                                                           torch.from_numpy(atoms_bonds[0])])
            data['atoms', 'interacts', 'atoms'].edge_index = torch.vstack([torch.from_numpy(np.arange(num_atoms, dtype='int64')), 
                                                                           torch.from_numpy(np.arange(num_atoms, dtype='int64'))])
            data['bonds', 'interacts', 'bonds'].edge_index = torch.vstack([torch.from_numpy(bond_idx), torch.from_numpy(bond_idx)])
            data.y = bonds_labels
            data.mid = k
            data_list.append(data)  
            # self.data, self.slices = self.collate(data_list)
        
        # collate all your processed data and save it into a single file
        torch.save(data_list, self.processed_paths[0])
 
    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]

##########################################################################################
################################ Collate Functions #######################################
##########################################################################################
def PYG_NC_collate(data):
    data_list = map(list, zip(*data))
    batch = Batch.from_data_list([add_self_loops(data)[0] for data in data_list])
    return batch
"""END HERE"""