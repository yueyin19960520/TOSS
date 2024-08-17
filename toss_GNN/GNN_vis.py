import plotly.graph_objects as go
from plotly.graph_objs import *
import pickle
import re
import os
import pandas as pd


class VIS():
    def __init__(self, prediction,):
        self.prediction = prediction
        
        openexcel = pd.read_excel('../pre_set.xlsx', sheet_name = "Radii_X")
        self.dic_s = openexcel.set_index("symbol").to_dict()["single"]
        dic_R = openexcel.set_index("symbol").to_dict()["R"]
        dic_G = openexcel.set_index("symbol").to_dict()["G"]
        dic_B = openexcel.set_index("symbol").to_dict()["B"]

        elements_list = openexcel["symbol"].tolist()
        self.vesta_color = {ele:"rgb"+str((dic_R[ele],dic_G[ele],dic_B[ele])) for ele in elements_list}

        scene = dict(xaxis = dict(showbackground=False, showgrid=False, showticklabels=False),
                     yaxis = dict(showbackground=False, showgrid=False, showticklabels=False),
                     zaxis = dict(showbackground=False, showgrid=False, showticklabels=False))
        
        self.draw()
            
    def draw(self,atom_ratio=0.3):
        struct = self.prediction["struct"]
        os_result = self.prediction["os"]
        connection = self.prediction["connection"]
        
        fig = go.Figure()

        raw_info = []
        for i,s in enumerate(struct.sites):
            ele = str(s.specie)
            features = list(s.coords) + [ele] + [self.dic_s[ele]] + [self.vesta_color[ele]] + [os_result[i]]
            raw_info.append(features)

        column_name = ["X","Y","Z","Element","size","color","valence"]
        df_info = pd.DataFrame(raw_info, columns=column_name)
        
        for i, coordinations in enumerate(connection):
            ele_i = struct.sites[i].specie.name
            i_xyz = list(struct.sites[i].coords)

            features_i = list(struct.sites[i].coords) + [ele_i] + [self.dic_s[ele_i]] + [self.vesta_color[ele_i]] + [ele_i+self.upper(os_result[i])]

            for j_xyz in coordinations:
                j = j_xyz[0]
                ele_j = struct.sites[j].specie.name
                
                features_j = j_xyz[1] + [ele_j] + [self.dic_s[ele_j]] + [self.vesta_color[ele_j]] + [ele_j+self.upper(os_result[j])]

                #################################################################################################
                mid_coords = [(i_xyz[0] + j_xyz[1][0])/2, (i_xyz[1] + j_xyz[1][1])/2, (i_xyz[2] + j_xyz[1][2])/2]
                i_mid = mid_coords + [ele_i] + [0.] + [self.vesta_color[ele_i]] + [""]
                j_mid = mid_coords + [ele_j] + [0.] + [self.vesta_color[ele_j]] + [""]
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
                            ))

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
                            ))
                
        scene = dict(xaxis = dict(showbackground=False, showgrid=False, showticklabels=False),
                     yaxis = dict(showbackground=False, showgrid=False, showticklabels=False),
                     zaxis = dict(showbackground=False, showgrid=False, showticklabels=False))        
        
        layout = Layout(height =600, width = 800, margin = dict(l=0, r=0, b=0, t=0), scene=scene)
        fig.update_layout(layout)
        fig.update_traces(showlegend=False)
        self.fig = fig
        
    def show_fig(self):
        self.fig.show()

    def save_fig(self,save_path=None):
        self.fig.write_html(save_path)       
    
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