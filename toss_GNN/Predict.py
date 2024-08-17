import torch
from pymatgen.core.structure import IStructure
from pymatgen.core.periodic_table import Element
import pandas as pd
import numpy as np
import scipy.sparse as sp
import copy
import os
from torch_geometric.data import HeteroData, Data


class Get_OS_by_models():
    def __init__(self, mid, LP_model, NC_model, server=None, filepath=None):
        path = os.path.dirname(os.path.dirname(__file__))
        self.server = server
        if self.server:
            self.structure_file = filepath
            self.preset_file =  os.path.join(path,"pre_set.xlsx")
        else:
            self.structure_file = "../structures/" + str(mid)
            self.preset_file = "../pre_set.xlsx"
        self.LP_model = LP_model
        self.NC_model = NC_model

        self.prefer_OS_ignore = {
            'Fe': 3, 'Co': 3, 'Ni': 2, 'Cu': 2, 'Ge': 2, 'As': 3, 'Se': 4,
            'Mo': 4, 'Tc': 4, 'Ru': 4, 'Rh': 3, 'Pd': 2, 'Ag': 1, 'Sb': 3, 'Te': 4,
            'W': 4, 'Re': 4, 'Os': 4, 'Ir': 4, 'Pt': 4, 'Au': 1, 'Tl': 1, 'Pb': 2,
            'Bi': 3, 'Po': 4, 'Sg': 4, 
            'Ce': 3, 'Pr': 3, 'Nd': 3, 'Tb': 3, 'Dy': 3, 
            'U': 4, 'Np': 5, 'Pu': 4, 'Am': 3, 'Cm': 3, 'Bk': 3, 'Cf': 3, 'Es': 3,
            'No': 2}
        
        self.prepare_dict()
        self.get_structure()

    def prepare_dict(self):
        openexcel = pd.read_excel(self.preset_file, sheet_name = "Radii_X")
        dic_s = openexcel.set_index("symbol").to_dict()["single"]
        dic_d = openexcel.set_index("symbol").to_dict()["double"]
        dic_t = openexcel.set_index("symbol").to_dict()["triple"]
        dic_x = openexcel.set_index("symbol").to_dict()["X"]
        openexcel = pd.read_excel(self.preset_file, sheet_name = "IP")

        self.pre_set = {}
        for k,v in dic_s.items():
            dict_temp = {'symbol':str(k), 'R1':float(dic_s[k]),'X':float(dic_x[k]), 'IP':openexcel[k].values.tolist(),
                         'R2':float(dic_d[k]), 'R3':float(dic_t[k])}
            self.pre_set.update({str(k):dict_temp})
            
        self.elements_list = list(self.pre_set.keys())
        
        for ele,os in self.prefer_OS_ignore.items():
            self.pre_set[ele]["IP"][os] = self.pre_set[ele]["IP"][os+1]-1

    def get_structure(self):
        self.struct = IStructure.from_file(self.structure_file)
        self.length_matrix = self.struct.distance_matrix

    def build_data_for_LP(self):
        mat = []
        for i in self.struct.sites:
            ele = str(i.specie.name)
            temp_dict = self.pre_set[ele]
            atom_num = self.elements_list.index(ele)+1
            row = [atom_num, temp_dict["X"], temp_dict["R1"], temp_dict["R2"], temp_dict["R3"]] + temp_dict["IP"][0:8]
            mat.append(row)

        mat = np.array(mat).astype("float32")
        atoms_data = pd.DataFrame(mat, columns=['Element','EN','R1','R2','R3','IP1','IP2','IP3','IP4','IP5','IP6','IP7','IP8'])

        num_atoms = atoms_data.shape[0]
        atoms_features = torch.from_numpy(atoms_data.to_numpy())  
        
        local = np.vstack(list(map(lambda x:np.where(x<1.5*(sorted(x)[1]),1,0),self.length_matrix)))
        trans_local = local.transpose()
        diff_local = trans_local - local
        mask = np.where(diff_local==1, 1, 0)
        global_adj = local + mask
        global_adj = sp.coo_matrix(global_adj)

        src = global_adj.row
        dst = global_adj.col
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

        length_list = list(map(lambda x:self.length_matrix[src_bond_dst[0,x]][src_bond_dst[2,x]], bond_idx))
        delta_EN_list = list(map(lambda x:abs(EN[src_bond_dst[0,x]]-EN[src_bond_dst[2,x]]), bond_idx))
        sum_R1_list = list(map(lambda x:abs(R1[src_bond_dst[0,x]]+R1[src_bond_dst[2,x]])*0.01,bond_idx))
        sum_R2_list = list(map(lambda x:abs(R2[src_bond_dst[0,x]]+R2[src_bond_dst[2,x]])*0.01,bond_idx))
        sum_R3_list = list(map(lambda x:abs(R3[src_bond_dst[0,x]]+R3[src_bond_dst[2,x]])*0.01,bond_idx))
        delta_IP1_list = list(map(lambda x:abs(IP1[src_bond_dst[0,x]]-IP1[src_bond_dst[2,x]])*0.001,bond_idx))
        delta_IP2_list = list(map(lambda x:abs(IP2[src_bond_dst[0,x]]-IP2[src_bond_dst[2,x]])*0.001,bond_idx))
        delta_IP3_list = list(map(lambda x:abs(IP3[src_bond_dst[0,x]]-IP3[src_bond_dst[2,x]])*0.001,bond_idx))
        ratio1_list = list(map(lambda x:(self.length_matrix[src_bond_dst[0,x]][src_bond_dst[2,x]]/abs(0.01*(R1[src_bond_dst[0,x]]+R1[src_bond_dst[2,x]]))),bond_idx))
        ratio2_list = list(map(lambda x:(self.length_matrix[src_bond_dst[0,x]][src_bond_dst[2,x]]/abs(0.01*(R2[src_bond_dst[0,x]]+R2[src_bond_dst[2,x]]))),bond_idx))
        ratio3_list = list(map(lambda x:(self.length_matrix[src_bond_dst[0,x]][src_bond_dst[2,x]]/abs(0.01*(R3[src_bond_dst[0,x]]+R3[src_bond_dst[2,x]]))),bond_idx))
        one_over_l_list = list(map(lambda x:1/(self.length_matrix[src_bond_dst[0,x]][src_bond_dst[2,x]]+1),bond_idx))
        square_l_list = list(map(lambda x:(self.length_matrix[src_bond_dst[0,x]][src_bond_dst[2,x]])**2,bond_idx))

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
        data.src = src
        data.dst = dst
        data.src_bond_dst = src_bond_dst
        data.bond_idx = bond_idx
        data.atoms_data = atoms_data
        return data
    
    def LP_predict(self):
        data = self.build_data_for_LP()

        self.LP_model.eval()
        bonds_preds = self.LP_model(data)

        num_atoms = data.atoms_data.shape[0]
        bonds_labels = np.array(list(map(lambda x:0. if x[0] < x[1] else 1.,bonds_preds.detach().numpy())))
        pred_adj = sp.coo_matrix((bonds_labels, (data.src, data.dst)),shape=(num_atoms, num_atoms))

        pair = list(filter(lambda x:x!=(None,None),
                           list(map(lambda x:(data.src_bond_dst[:,x][0],data.src_bond_dst[:,x][2]) if bonds_labels[x] == 1 else (None,None),
                                    data.bond_idx))))

        shell_idx_list = list(map(lambda y:list(filter(lambda x:x!=None,
                                                  list(map(lambda x:x[1] if x[0] == y else None, 
                                                           pair)))), np.arange(num_atoms)))
        return pred_adj.todense(), shell_idx_list, data

    def _get_image(self, i, j):
        sites = self.struct.sites
        image_list = [[-1, -1, -1],[-1, -1, 0],[-1, -1, 1],[-1, 0, -1],[-1, 0, 0],[-1, 0, 1],[-1, 1, -1],[-1, 1, 0],
                      [-1, 1, 1],[0, -1, -1],[0, -1, 0],[0, -1, 1],[0, 0, -1],[0, 0, 0],[0, 0, 1],[0, 1, -1],[0, 1, 0], 
                      [0, 1, 1],[1, -1, -1],[1, -1, 0],[1, -1, 1],[1, 0, -1],[1, 0, 0],[1, 0, 1],[1, 1, -1],[1, 1, 0],[1, 1, 1]]
        IMAGE = []
        for image in image_list:
            if round(sites[i].distance(sites[j],jimage = image),4) <= round(self.length_matrix[i][j], 4):
                IMAGE.append(image)

        J_coords = copy.deepcopy(IMAGE)
        j_coords = [sites[j].a, sites[j].b, sites[j].c] 

        for image in J_coords:
            for c in [0,1,2]:
                image[c] += j_coords[c] 

        J_coords = [self.struct.lattice.get_cartesian_coords(image) for image in J_coords]
        return IMAGE, J_coords
    
    def _apply_images(self, shell_idx_list):
        shell_idx_list_imaged = []
        for i,j_list in enumerate(shell_idx_list):
            temp_shell_idx_list = []
            sub_shell_idx_list = []
            for j in j_list:
                IMAGE,J_coords = self._get_image(i,j)
                for image,j_coords in zip(IMAGE,J_coords):
                    temp_shell_idx_list.append((j,list(j_coords)))
                    sub_shell_idx_list.append(j)
            shell_idx_list_imaged.append(temp_shell_idx_list)
        return shell_idx_list_imaged
        
    def build_data_for_NC(self):
        pred_adj, shell_idx_list, LP_data = self.LP_predict()
        connection = self._apply_images(shell_idx_list)

        shell_CN_list = list(map(lambda x:len(x), connection))
        shell_idx_list = list(map(lambda x:list(map(lambda y:y[0], x)), connection))
        shell_SEN_list = list(map(lambda x:sum(list(map(lambda y:self.pre_set[str(self.struct.sites[y].specie.name)]["X"],x))), shell_idx_list))
        
        LP_data.atoms_data.insert(loc=2, column="CN", value = np.array(shell_CN_list).astype("float32"))
        LP_data.atoms_data.insert(loc=3, column="SEN",value = np.array(shell_SEN_list).astype("float32"))

        node_features = torch.from_numpy(LP_data.atoms_data.to_numpy())
        edge_index = torch.from_numpy(np.array([sp.coo_matrix(pred_adj).row, sp.coo_matrix(pred_adj).col])).long()
        data = Data(x=node_features, edge_index=edge_index)
        data.shell_CN_list = shell_CN_list
        data.connection = connection
        return data
        
    def NC_predict(self):
        data = self.build_data_for_NC()
        self.NC_model.eval()
        pred = self.NC_model(data).detach().numpy()
        one_hot = list(map(lambda x:np.where(x==max(x),1,0), pred))
        element_list = list(map(lambda s:s.specie.name, self.struct.sites))
        OS = list(map(lambda x:list(np.arange(-4,8))[list(x).index(1)], one_hot))

        if self.server:
            result = pd.DataFrame(np.vstack([np.array(element_list),np.array(OS),np.array(data.shell_CN_list)]))
            result.index = ["Elements", "Valence","Coordination Number"]
            return result
        else:
            shell_CN_list = list(map(lambda x:len(x), data.connection))
            result = {"struct":self.struct, "connection":data.connection, 
                      "os":OS, "cn":shell_CN_list, "ele":element_list}
            return result