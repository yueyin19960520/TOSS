from model_utils_dgl import *
import torch
import pickle
from pymatgen.core.structure import IStructure
from pymatgen.core.periodic_table import Element
from pymatgen.transformations.standard_transformations import SupercellTransformation
import pandas as pd
import numpy as np
import scipy.sparse as sp
import dgl
import copy
import plotly.graph_objects as go
from plotly.graph_objs import *
import os


class vis_LP_from_cif():
    path = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]    #D:/share/reTOSS/
    def __init__(self,  model, mid, max_bond_length=3, supercell=False, path=path):
        #path = os.path.split(os.path.dirname(__file__))[0]
        self.preset_file = path + "/pre_set.xlsx"
        self.structure_file = path + "/structures/" + str(mid)
        self.model = model
        self.max_bond_length = max_bond_length
        self.supercell = supercell


    def _prepare_dict(self):
        openexcel = pd.read_excel(self.preset_file, sheet_name = "Radii_X")
        dic_s = openexcel.set_index("symbol").to_dict()["single"]
        dic_d = openexcel.set_index("symbol").to_dict()["double"]
        dic_t = openexcel.set_index("symbol").to_dict()["triple"]
        dic_R = openexcel.set_index("symbol").to_dict()["R"]
        dic_G = openexcel.set_index("symbol").to_dict()["G"]
        dic_B = openexcel.set_index("symbol").to_dict()["B"]
        dic_x = openexcel.set_index("symbol").to_dict()["X"]
        openexcel = pd.read_excel(self.preset_file, sheet_name = "IP")

        self.pre_set = {}
        for k,v in dic_s.items():
            dict_temp = {}
            covalent_radius = float(dic_s[k])
            second_covalent_radius = float(dic_d[k])
            third_covalent_radius = float(dic_t[k])
            X = float(dic_x[k])
            symbol = str(k)
            ele = Element(k)
            list_IP = openexcel[k].values.tolist()
            R = dic_R[k]
            G = dic_G[k]
            B = dic_B[k]
            dict_temp = {'symbol':symbol, 
                         'covalent_radius':covalent_radius,
                         'X':X, 
                         'IP':list_IP,
                         'second_covalent_radius':second_covalent_radius, 
                         'third_covalent_radius':third_covalent_radius,
                         'R':R,
                         'G':G,
                         'B':B}
            self.pre_set.update({symbol:dict_temp})
            
        self.elements_list = list(self.pre_set.keys())
        self.vesta_color = {ele:"rgb"+str((dic_R[ele],dic_G[ele],dic_B[ele])) for ele in self.elements_list}


    def _get_structure(self):
        self.struct = IStructure.from_file(self.structure_file)
        self.length_matrix = self.struct.distance_matrix

        if self.supercell:
            ST = SupercellTransformation(scaling_matrix=((3, 0, 0), (0, 3, 0), (0, 0, 3)))
            self.struct = ST.apply_transformation(self.struct)
            self.length_matrix = self.struct.distance_matrix


    def _build_graph_for_pred(self):
        #elements_list = list(self.pre_set.keys())
        mat = []
        for i in self.struct.sites:
            ele = str(i.specie)
            temp_dict = self.pre_set[ele]
            atom_num = self.elements_list.index(ele)+1
            EN = temp_dict["X"]
            R1 = temp_dict["covalent_radius"]
            R2 = temp_dict["second_covalent_radius"]
            R3 = temp_dict["third_covalent_radius"]
            IP_list = temp_dict["IP"][0:8]
            row = [atom_num, EN, R1, R2, R3] + IP_list
            mat.append(row)

        mat = np.array(mat).astype("float32")
        self.atoms_data = pd.DataFrame(mat, columns=['Element','EN','R1','R2','R3','IP1','IP2','IP3','IP4','IP5','IP6','IP7','IP8'])

        self.num_atoms = self.atoms_data.shape[0]
        atoms_features = torch.from_numpy(self.atoms_data.to_numpy())    

        global_adj = sp.coo_matrix(np.ones((self.num_atoms, self.num_atoms)))
        global_adj = sp.coo_matrix(np.where(self.length_matrix <= self.max_bond_length, 1, 0))
        if self.supercell:
            global_adj = sp.coo_matrix(np.where(self.length_matrix <= super_bond_length, 1, 0))

        self.src = global_adj.row
        self.dst = global_adj.col
        self.num_bonds = len(self.src)
        self.bond_idx = np.arange(self.num_bonds, dtype="int32")
        self.src_bond_dst = np.vstack((self.src, self.bond_idx, self.dst))
        atoms_bonds = np.vstack((np.hstack((self.src_bond_dst[0], self.src_bond_dst[-1])),np.hstack((self.bond_idx, self.bond_idx))))

        graph_data = {('atoms', 'interacts', 'bonds'): (torch.from_numpy(atoms_bonds[0]), 
                                                        torch.from_numpy(atoms_bonds[1])),
                      ('bonds', 'interacts', 'atoms'): (torch.from_numpy(atoms_bonds[1]),
                                                        torch.from_numpy(atoms_bonds[0]))}

        EN = list(self.atoms_data["EN"])
        R1 = list(self.atoms_data["R1"])
        R2 = list(self.atoms_data["R2"])
        R3 = list(self.atoms_data["R3"])
        IP1 = list(self.atoms_data["IP1"])
        IP2 = list(self.atoms_data["IP2"])
        IP3 = list(self.atoms_data["IP3"])

        length_list = list(map(lambda x:self.length_matrix[self.src_bond_dst[0,x]][self.src_bond_dst[2,x]], self.bond_idx))
        delta_EN_list = list(map(lambda x:abs(EN[self.src_bond_dst[0,x]]-EN[self.src_bond_dst[2,x]]), self.bond_idx))
        sum_R1_list = list(map(lambda x:abs(R1[self.src_bond_dst[0,x]]+R1[self.src_bond_dst[2,x]])*0.01,self.bond_idx))
        sum_R2_list = list(map(lambda x:abs(R2[self.src_bond_dst[0,x]]+R2[self.src_bond_dst[2,x]])*0.01,self.bond_idx))
        sum_R3_list = list(map(lambda x:abs(R3[self.src_bond_dst[0,x]]+R3[self.src_bond_dst[2,x]])*0.01,self.bond_idx))
        delta_IP1_list = list(map(lambda x:abs(IP1[self.src_bond_dst[0,x]]-IP1[self.src_bond_dst[2,x]])*0.001,self.bond_idx))
        delta_IP2_list = list(map(lambda x:abs(IP2[self.src_bond_dst[0,x]]-IP2[self.src_bond_dst[2,x]])*0.001,self.bond_idx))
        delta_IP3_list = list(map(lambda x:abs(IP3[self.src_bond_dst[0,x]]-IP3[self.src_bond_dst[2,x]])*0.001,self.bond_idx))
        ratio1_list = list(map(lambda x:(self.length_matrix[self.src_bond_dst[0,x]][self.src_bond_dst[2,x]]/abs(0.01*(R1[self.src_bond_dst[0,x]]+R1[self.src_bond_dst[2,x]]))),self.bond_idx))
        ratio2_list = list(map(lambda x:(self.length_matrix[self.src_bond_dst[0,x]][self.src_bond_dst[2,x]]/abs(0.01*(R2[self.src_bond_dst[0,x]]+R2[self.src_bond_dst[2,x]]))),self.bond_idx))
        ratio3_list = list(map(lambda x:(self.length_matrix[self.src_bond_dst[0,x]][self.src_bond_dst[2,x]]/abs(0.01*(R3[self.src_bond_dst[0,x]]+R3[self.src_bond_dst[2,x]]))),self.bond_idx))
        one_over_l_list = list(map(lambda x:1/(self.length_matrix[self.src_bond_dst[0,x]][self.src_bond_dst[2,x]]+1),self.bond_idx))
        square_l_list = list(map(lambda x:(self.length_matrix[self.src_bond_dst[0,x]][self.src_bond_dst[2,x]])**2,self.bond_idx))

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
        self.graph = dgl.heterograph(graph_data)
        self.graph.nodes["atoms"].data['h'] = atoms_features
        self.graph.nodes["bonds"].data["h"] = bonds_features
    
    def _do_predict(self):
        self._prepare_dict()
        self._get_structure()
        self._build_graph_for_pred()
        
        atoms_feats = self.graph.ndata["h"]["atoms"]
        bonds_feats = self.graph.ndata["h"]["bonds"]
        self.model.eval()
        bonds_preds = self.model(self.graph, {"atoms":atoms_feats, "bonds":bonds_feats})
        
        bonds_labels = np.array(list(map(lambda x:0. if x[0] < x[1] else 1.,bonds_preds.detach().numpy())))
        pred_adj = sp.coo_matrix((bonds_labels, (self.src, self.dst)),shape=(self.num_atoms, self.num_atoms))

        pair = list(filter(lambda x:x!=(None,None),
                           list(map(lambda x:(self.src_bond_dst[:,x][0],self.src_bond_dst[:,x][2]) if bonds_labels[x] == 1 else (None,None),
                                    self.bond_idx))))

        shell_idx = list(map(lambda y:list(filter(lambda x:x!=None,
                                                  list(map(lambda x:x[1] if x[0] == y else None, 
                                                           pair)))), 
                             np.arange(self.num_atoms)))
        
        self.tot_num_bonds = bonds_labels.sum()
        self.pred_adj = pred_adj.todense()
        self.shell_idx_list = shell_idx


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

        
    def _apply_images(self):
        self.plotting_coordinations = []
        self.shell_idx_list_with_images = []
        for i,j_list in enumerate(self.shell_idx_list):
            temp_shell_idx_list = []
            sub_shell_idx_list = []
            for j in j_list:
                IMAGE,J_coords = self._get_image(i,j)
                for image,j_coords in zip(IMAGE,J_coords):
                    temp_shell_idx_list.append((j,list(j_coords)))
                    sub_shell_idx_list.append(j)
            self.plotting_coordinations.append(temp_shell_idx_list)
            self.shell_idx_list_with_images.append(sub_shell_idx_list)
        
            
    
    def draw(self, atom_ratio=0.35, line_width=12, row = None, col = None):
        fig = go.Figure()
        self._do_predict()
        self._apply_images()
        raw_info = []
        for i,s in enumerate(self.struct.sites):
            ele = str(s.specie)
            features = list(s.coords) + [ele] + [self.pre_set[ele]["covalent_radius"]] + [self.vesta_color[ele]]
            raw_info.append(features)

        column_name = ["X","Y","Z","Element","size","color"]
        df_info = pd.DataFrame(raw_info, columns=column_name)

        for i, coordinations in enumerate(self.plotting_coordinations):
            ele_i = str(self.struct.sites[i].specie)
            i_xyz = list(self.struct.sites[i].coords)
            features_i = i_xyz + [ele_i] + [self.pre_set[ele_i]["covalent_radius"]] + [self.vesta_color[ele_i]]

            for j_xyz in coordinations:
                j = j_xyz[0]
                ele_j = str(self.struct.sites[j].specie)
                features_j = j_xyz[1] + [ele_j] + [self.pre_set[ele_j]["covalent_radius"]] + [self.vesta_color[ele_j]]
                features_j = j_xyz[1] + [ele_j] + [line_width/atom_ratio/2] + [self.vesta_color[ele_j]]
                
                ##############################
                mid_coords = [(i_xyz[0] + j_xyz[1][0])/2, (i_xyz[1] + j_xyz[1][1])/2, (i_xyz[2] + j_xyz[1][2])/2]
                i_mid = mid_coords + [ele_i] + [0.] + [self.vesta_color[ele_i]]
                j_mid = mid_coords + [ele_j] + [0.] + [self.vesta_color[ele_j]]
                ##############################

                temp_info = [features_i, i_mid]
                temp_df_info = pd.DataFrame(temp_info, columns=column_name)

                fig.add_trace(go.Scatter3d( x=temp_df_info["X"], y=temp_df_info["Y"], z=temp_df_info["Z"],
                            mode = "lines+markers+text",
                            marker = dict(size = temp_df_info["size"] * atom_ratio, opacity = 1, color = temp_df_info["color"]),
                            line = dict(color=temp_df_info["color"], width = line_width, cauto = False, autocolorscale=False),
                            hoverinfo = "skip"), row=row, col=col)
                
                temp_info = [j_mid, features_j]
                temp_df_info = pd.DataFrame(temp_info, columns=column_name)

                fig.add_trace(go.Scatter3d( x=temp_df_info["X"], y=temp_df_info["Y"], z=temp_df_info["Z"],
                            mode = "lines+markers+text",
                            marker = dict(size = temp_df_info["size"] * atom_ratio, opacity = 1, color = temp_df_info["color"]),
                            line = dict(color=temp_df_info["color"], width = line_width, cauto = False, autocolorscale=False),
                            hoverinfo = "skip"), row=row, col=col)
                
        scene = dict(xaxis = dict(showbackground=False, showgrid=False, showticklabels=False),
                     yaxis = dict(showbackground=False, showgrid=False, showticklabels=False),
                     zaxis = dict(showbackground=False, showgrid=False, showticklabels=False))
        layout = Layout(height =600, width = 800, margin = dict(l=0, r=0, b=0, t=0), scene = scene)
        fig.update_layout(layout)
        fig.update_traces(showlegend=False)
        return fig

    def os_predict(self, model):
        self._do_predict()
        self._apply_images()
        self.shell_CN_list = list(map(lambda x:len(x), self.plotting_coordinations))
        self.shell_idx_list = list(map(lambda x:list(map(lambda y:y[0], x)),self.plotting_coordinations))
        sites = self.struct.sites
        self.shell_SEN_list = list(map(lambda x:sum(list(map(lambda y:self.pre_set[str(sites[y].specie)]["X"],x))), self.shell_idx_list))
        
        self.atoms_data.insert(loc=2, column="CN", value = np.array(self.shell_CN_list).astype("float32"))
        self.atoms_data.insert(loc=3, column="SEN",value = np.array(self.shell_SEN_list).astype("float32"))

        node_features = torch.from_numpy(self.atoms_data.to_numpy())
        src = sp.coo_matrix(self.pred_adj).row
        dst = sp.coo_matrix(self.pred_adj).col
        
        graph = dgl.graph((src, dst), num_nodes=node_features.shape[0])
        graph = dgl.add_self_loop(graph)
        graph.ndata['h'] = node_features

        model.eval()
        self.pred = model(graph, graph.ndata.pop('h')).detach().numpy()
        self.prob = torch.softmax(torch.from_numpy(self.pred), dim=1)
        self.one_hot = list(map(lambda x:np.where(x==max(x), 1,0), self.pred))
        OS = list(map(lambda x:list(np.arange(-4,8))[list(x).index(1)], self.one_hot))
        return OS
"""END HERE"""