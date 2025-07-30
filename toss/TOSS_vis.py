import plotly.graph_objects as go
from plotly.graph_objs import *
from plotly.subplots import make_subplots
from post_process import *
import pandas as pd
from get_fos import GET_FOS
from post_process import *
from result import RESULT
from tune import TUNE
import time
import pickle
import re
import os


toss_path = os.path.dirname(os.path.dirname(__file__))
n_loop = max([int(f.split('_')[-1].split('.')[0]) for f in list(filter(lambda f:f if "_loop_" in f else None, os.listdir(toss_path)))])

file_get= open(os.path.join(toss_path, "global_normalized_normed_dict_loop_%s.pkl"%n_loop),"rb")
global_normalized_normed_dict = pickle.load(file_get)
file_get.close()

file_get= open(os.path.join(toss_path, "global_mean_dict_loop_%s.pkl"%n_loop),"rb")
global_mean_dict = pickle.load(file_get)
file_get.close()

file_get= open(os.path.join(toss_path, "global_sigma_dict_loop_%s.pkl"%n_loop),"rb")
global_sigma_dict = pickle.load(file_get)
file_get.close()



###THE input of the class is the res###
class VIS():
    def __init__(self, 
                 res, 
                 old_res=None,
                 global_nomalized_normed_dict=global_normalized_normed_dict, 
                 global_mean_dict=global_mean_dict, 
                 global_sigma_dict=global_sigma_dict,
                 loss_ratio = 10, 
                 atom_ratio = 0.3, 
                 loss_opacity = 0.3,
                 height=1080,
                 width=1920): 
        self.height = height
        self.width = width

        openexcel = pd.read_excel(os.path.join(toss_path,'pre_set.xlsx'), sheet_name = "Radii_X")
        dic_R = openexcel.set_index("symbol").to_dict()["R"]
        dic_G = openexcel.set_index("symbol").to_dict()["G"]
        dic_B = openexcel.set_index("symbol").to_dict()["B"]

        elements_list = openexcel["symbol"].tolist()
        vesta_color = {ele:"rgb"+str((dic_R[ele],dic_G[ele],dic_B[ele])) for ele in elements_list}

        scene = dict(xaxis = dict(showbackground=False, showgrid=False, showticklabels=False),
                     yaxis = dict(showbackground=False, showgrid=False, showticklabels=False),
                     zaxis = dict(showbackground=False, showgrid=False, showticklabels=False))
    
        if res.shell_CN_list == old_res.shell_CN_list and res.initial_vl == res.final_vl:
            print("CNs are SAME and OSs are SAME ! USE one fig !")

            fig = go.Figure()

            temp_pair_info_per_atom  = self.spider_pair_length_with_CN_unnorm_per_atom(res.initial_vl, res)

            initial_loss_list = self.cal_loss_func_by_MAP_per_atom(temp_pair_info_per_atom, 
                                                                   global_nomalized_normed_dict, 
                                                                   global_mean_dict, 
                                                                   global_sigma_dict)

            plotting_coordinations = self.convert_images_to_coordinations(res)

            fig = self.draw(res, res.initial_vl, initial_loss_list, plotting_coordinations, fig, vesta_color, 
                loss_ratio = loss_ratio, atom_ratio = atom_ratio, loss_opacity = loss_opacity, row = None, col = None)

            layout = Layout(height = self.height, width = self.width, margin = dict(l=0, r=0, b=0, t=0), scene = scene)


        elif res.shell_CN_list != old_res.shell_CN_list:
            print("CNs are DIFFERENT! USE two figs !")

            fig = make_subplots(rows=1, cols=2,specs = [[{"type":"scatter3D"}, {"type":"scatter3D"}]])

            temp_pair_info_per_atom  = self.spider_pair_length_with_CN_unnorm_per_atom(old_res.sum_of_valence, old_res)

            initial_loss_list = self.cal_loss_func_by_MAP_per_atom(temp_pair_info_per_atom, 
                                                                   global_nomalized_normed_dict, 
                                                                   global_mean_dict, 
                                                                   global_sigma_dict)

            pauling_ratio = 1.2**(len(old_res.species_uni_list)-len(res.species_uni_list)) 
            initial_loss_list = [l*pauling_ratio for l in initial_loss_list]

            plotting_coordinations = self.convert_images_to_coordinations(old_res)

            fig = self.draw(old_res, 
                            old_res.sum_of_valence, 
                            initial_loss_list, 
                            plotting_coordinations, 
                            fig, 
                            vesta_color, 
                            loss_ratio = loss_ratio, 
                            atom_ratio = atom_ratio, 
                            loss_opacity = loss_opacity, 
                            row = 1, 
                            col = 1)


            temp_pair_info_per_atom  = self.spider_pair_length_with_CN_unnorm_per_atom(res.final_vl, res)
            final_loss_list = self.cal_loss_func_by_MAP_per_atom(temp_pair_info_per_atom, 
                                                                 global_nomalized_normed_dict, 
                                                                 global_mean_dict, 
                                                                 global_sigma_dict)

            plotting_coordinations = self.convert_images_to_coordinations(res)

            fig = self.draw(res, 
                            res.final_vl, 
                            final_loss_list, 
                            plotting_coordinations, 
                            fig, 
                            vesta_color, 
                            loss_ratio = loss_ratio, 
                            atom_ratio = atom_ratio, 
                            loss_opacity = loss_opacity, 
                            row = 1, 
                            col = 2)

            layout = Layout(height = self.height, width = self.width, margin = dict(l=0, r=0, b=0, t=0), scene1 = scene, scene2 = scene)

        elif res.shell_CN_list == old_res.shell_CN_list and res.initial_vl != res.final_vl:
            print("CNs are SAME! OSs are DIFFERENT! USE two figs !")

            fig = make_subplots(rows=1, cols=2,specs = [[{"type":"scatter3D"}, {"type":"scatter3D"}]])

            temp_pair_info_per_atom  = self.spider_pair_length_with_CN_unnorm_per_atom(res.initial_vl, res)
            initial_loss_list = self.cal_loss_func_by_MAP_per_atom(temp_pair_info_per_atom, 
                                                                   global_nomalized_normed_dict, 
                                                                   global_mean_dict, 
                                                                   global_sigma_dict)

            pauling_ratio = 1.2**(len(res.species_uni_list)-len(res.species_uni_list))
            initial_loss_list = [l*pauling_ratio for l in initial_loss_list]
            
            plotting_coordinations = self.convert_images_to_coordinations(res)

            fig = self.draw(res, 
                            res.initial_vl, 
                            initial_loss_list, 
                            plotting_coordinations, 
                            fig, 
                            vesta_color, 
                            loss_ratio = loss_ratio, 
                            atom_ratio = atom_ratio, 
                            loss_opacity = loss_opacity, 
                            row = 1, 
                            col = 1)


            temp_pair_info_per_atom  = self.spider_pair_length_with_CN_unnorm_per_atom(res.final_vl, res)
            final_loss_list = self.cal_loss_func_by_MAP_per_atom(temp_pair_info_per_atom, 
                                                                 global_nomalized_normed_dict, 
                                                                 global_mean_dict, 
                                                                 global_sigma_dict)
            fig = self.draw(res, 
                            res.final_vl, 
                            final_loss_list, 
                            plotting_coordinations, 
                            fig, 
                            vesta_color, 
                            loss_ratio = loss_ratio, 
                            atom_ratio = atom_ratio, 
                            loss_opacity = loss_opacity, 
                            row = 1, 
                            col = 2)

            layout = Layout(height = self.height, width = self.width, margin = dict(l=0, r=0, b=0, t=0), scene1 = scene, scene2 = scene)
        
        fig.update_layout(layout)
        fig.update_traces(showlegend=False)
        self.fig = fig

    def show_fig(self):
        self.fig.show()

    def save_fig(self,save_path=None):
        self.fig.write_html(save_path)

    def convert_images_to_coordinations(self, res):
        plotting_coordinations = []
        for images_dict in res.SHELL_idx_list_with_images:
            temp_shell_idx_list = []
            for j,image_list in images_dict.items():
                for image in image_list:
                    ori_relative_coords = [res.sites[j].a,res.sites[j].b,res.sites[j].c]
                    img_relative_coords = [ori_relative_coords[x] + image[x] for x in [0,1,2]]
                    img_cartesian_coords = res.struct.lattice.get_cartesian_coords(img_relative_coords)
                    temp_shell_idx_list.append((j,list(img_cartesian_coords)))
            plotting_coordinations.append(temp_shell_idx_list)
        return plotting_coordinations

    def draw(self, res, valence_list, loss_list, plotting_coordinations, fig, default_color_dict, 
        loss_ratio, atom_ratio, loss_opacity, row = None, col = None):

        raw_info = []
        for i,s in enumerate(res.sites):
            ele = s.specie.name
            features = list(s.coords) + \
                            [ele] + \
                            [res.dict_ele[ele]["covalent_radius"]] + \
                            [default_color_dict[ele]] + \
                            [ele+self.upper(valence_list[i])] + \
                            [round(loss_list[i],2)]

            raw_info.append(features)

        column_name = ["X","Y","Z","Element","size","color","valence","LOSS"]
        df_info = pd.DataFrame(raw_info, columns=column_name)
        
        connection = plotting_coordinations
        
        fig.add_trace(go.Scatter3d(
            x=df_info["X"], 
            y=df_info["Y"], 
            z=df_info["Z"],
            mode = "markers",
            marker = dict(size = df_info["LOSS"] * loss_ratio, opacity = loss_opacity, color = df_info["color"]),
            hoverinfo = "skip"
            ), row=row, col=col)
        
        for i, coordinations in enumerate(connection):
            ele_i = res.sites[i].specie.name
            i_xyz = list(res.sites[i].coords)

            features_i = list(res.sites[i].coords) + \
                              [ele_i] + \
                              [res.dict_ele[ele_i]["covalent_radius"]] + \
                              [default_color_dict[ele_i]] + \
                              [ele_i+self.upper(valence_list[i])] + \
                              [round(loss_list[i],2)]

            for j_xyz in coordinations:
                j = j_xyz[0]
                ele_j = res.sites[j].specie.name
                features_j = j_xyz[1] + \
                             [ele_j] + \
                             [res.dict_ele[ele_j]["covalent_radius"]] + \
                             [default_color_dict[ele_j]] + \
                             [ele_j+self.upper(valence_list[j])] + \
                             [round(loss_list[j],2)]

                #################################################################################################
                mid_coords = [(i_xyz[0] + j_xyz[1][0])/2, (i_xyz[1] + j_xyz[1][1])/2, (i_xyz[2] + j_xyz[1][2])/2]
                i_mid = mid_coords + [ele_i] + [0.] + [default_color_dict[ele_i]] + [""] + [0.]
                j_mid = mid_coords + [ele_j] + [0.] + [default_color_dict[ele_j]] + [""] + [0.]
                #################################################################################################

                temp_info = [features_i, i_mid]
                temp_df_info = pd.DataFrame(temp_info, columns=column_name)
                
                fig.add_trace(go.Scatter3d(
                            x=temp_df_info["X"], 
                            y=temp_df_info["Y"], 
                            z=temp_df_info["Z"],
                            mode = "lines+markers+text",
                            marker = dict(size = temp_df_info["size"] * atom_ratio, opacity = 1, color = temp_df_info["color"]),
                            line = dict(color=temp_df_info["color"], width = 15,cauto = False,autocolorscale=False),
                            text = temp_df_info["valence"],
                            hoverinfo = "skip"
                            ), row=row, col=col)

                temp_info = [j_mid, features_j]
                temp_df_info = pd.DataFrame(temp_info, columns=column_name)
                
                fig.add_trace(go.Scatter3d(
                            x=temp_df_info["X"], 
                            y=temp_df_info["Y"], 
                            z=temp_df_info["Z"],
                            mode = "lines+markers+text",
                            marker = dict(size = temp_df_info["size"] * atom_ratio, opacity = 1, color = temp_df_info["color"]),
                            line = dict(color=temp_df_info["color"], width = 15,cauto = False,autocolorscale=False),
                            text = temp_df_info["valence"],
                            hoverinfo = "skip"
                            ), row=row, col=col)
        
        return fig

    def upper(self,v):
        upper_dict = {"0":'\u2070',"1":'\u00B9',"2":'\u00B2',"3":'\u00B3',"4":'\u2074',
                      "5":'\u2075',"6":'\u2076',"7":'\u2077',"8":'\u2078',"9":'\u2079',
                      "+":'\u207A',"-":'\u207B'}
        if v > 0:    
            return upper_dict[str(v)] + upper_dict["+"]
        elif v < 0:
            return upper_dict[str(abs(v))] + upper_dict["-"]
        else:
            return upper_dict["0"]


    def spider_pair_length_with_CN_unnorm_per_atom(self, valence_list, res):
        temp_pair_info_per_atom = []
        for i in res.idx:
            length_list = res.matrix_of_length[i]
            temp_pair_info_per_j = []
            for j in res.shell_idx_list[i]:
                ele_c = get_ele_from_sites(i,res)
                ele_n = get_ele_from_sites(j,res)
                v_c = str(valence_list[i])
                v_n = str(valence_list[j])
                CN_c = len(res.shell_ele_list[i])
                CN_n = len(res.shell_ele_list[j])
                
                if res.periodic_table.elements_list.index(ele_c) < res.periodic_table.elements_list.index(ele_n):
                    pair_name = (ele_c, ele_n)
                    pair_CN = (CN_c, CN_n)
                    pair_OS = (v_c, v_n)
                if res.periodic_table.elements_list.index(ele_c) > res.periodic_table.elements_list.index(ele_n):
                    pair_name = (ele_n, ele_c)
                    pair_CN = (CN_n, CN_c)
                    pair_OS = (v_n, v_c)
                if res.periodic_table.elements_list.index(ele_c) == res.periodic_table.elements_list.index(ele_n):
                    if v_c <= v_n:
                        pair_name = (ele_c, ele_n)
                        pair_CN = (CN_c, CN_n)
                        pair_OS = (v_c, v_n)
                    else:
                        pair_name = (ele_n, ele_c)
                        pair_CN = (CN_n, CN_c)
                        pair_OS = (v_n, v_c)

                CN_name = pair_CN
                OS_name = pair_OS
                label = (CN_name, OS_name)
                
                temp_pair_info_per_j.append((pair_name,label,length_list[j]))     
            temp_pair_info_per_atom.append(temp_pair_info_per_j)
        return temp_pair_info_per_atom


    def cal_loss_func_by_MAP_per_atom(self, 
                                      temp_pair_info_per_atom, 
                                      pred_dict, 
                                      global_sigma_dict, 
                                      global_mean_dict):
        loss_per_atom = []
        
        for atom_info in temp_pair_info_per_atom:
            likelyhood = 0
            prior = 0
            for j_atom_info in atom_info:
                pair_name = j_atom_info[0]
                label = j_atom_info[1]
                l = j_atom_info[2]
                
                if pair_name in pred_dict:
                    useful_pair = pred_dict[pair_name]
                    # Prior
                    NL = sum([v[1] for k,v in useful_pair.items() if k[0] == label[0]])
                    if NL == 0:
                        NL = sum([v[1] for k,v in useful_pair.items()])
                    try:
                        nl = useful_pair[label][1]
                    except Exception as e:
                        nl = 1
                    prior += math.log(nl/NL)

                    # Likelyhood
                    key = (pair_name, label[0], label[1])
                    try:
                        mean = round(global_mean_dict[key],3)
                        sigma = round(global_sigma_dict[key],3)
                        sigma = 0.01 if sigma == 0 else sigma
                    except Exception as e:
                        possible_keys = [k for k in global_mean_dict.keys() if k[0] == pair_name]
                        mean = np.mean([global_mean_dict[key] for key in possible_keys])
                        sigma = np.mean([global_sigma_dict[key] for key in possible_keys])

                    gx = (1/(np.sqrt(2*np.pi)*sigma)) * np.exp(-(round(l,3)-mean)**2/(2*sigma**2))
                    gx_den = gx * 0.001
                    math_domin_limit = 10**(-323.60)
                    gx_den = gx_den if gx_den > math_domin_limit else math_domin_limit
                    likelyhood += math.log(gx_den)
                else:
                    prior += math.log(1/100000)
                    raise ValueError("WRONG!")
            loss_per_atom.append(prior + likelyhood)

        return list(map(lambda x:-1*x[0]/len(x[1]),zip(loss_per_atom,temp_pair_info_per_atom)))