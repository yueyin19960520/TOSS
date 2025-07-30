import pandas as pd
from pymatgen.core.periodic_table import Element
from collections import Counter
import numpy as np  
import os


prefer_OS_ignore_max= {'Fe':3, 'Co':3, 'Ni':2, 'Cu':2, 'Ge':2, 'As':3, 'Se':4,
                       'Mo':4, 'Tc':4, 'Ru':4, 'Rh':3, 'Pd':2, 'Ag':1, 'Sb':3, 'Te':4,
                       'W' :4, 'Re':4, 'Os':4, 'Ir':4, 'Pt':4, 'Au':1, 'Tl':1, 'Pb':2, 'Bi':3, 'Po':4,
                       'Sg':4, 
                       'Ce':3, 'Pr':3, 'Nd':3, 'Tb':3, 'Dy':3, 
                       'U' :4, 'Np':5, 'Pu':4, 'Am':3, 'Cm':3, 'Bk':3, 'Cf':3, 'Es':3,'No':2}

path = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]

class PRE_SET():

    def __init__(self, spider = False, work_type = None):
        #make up the dictionaries for different covalent matrix and the Ognove's Electron nagetivities.
        openexcel = pd.read_excel(path + '/pre_set.xlsx', sheet_name = "Radii_X")     #switch the ionic and covalent radius.
        #openexcel = pd.read_excel('The Ionic Radius of elements.xlsx')
        dic_s = openexcel.set_index("symbol").to_dict()["single"]
        dic_d = openexcel.set_index("symbol").to_dict()["double"]
        dic_t = openexcel.set_index("symbol").to_dict()["triple"]
        dic_x = openexcel.set_index("symbol").to_dict()["X"]
        dic_R = openexcel.set_index("symbol").to_dict()["R"]
        dic_G = openexcel.set_index("symbol").to_dict()["G"]
        dic_B = openexcel.set_index("symbol").to_dict()["B"]

        openexcel = pd.read_excel(path + '/pre_set.xlsx', sheet_name = "IP")
        temp_dict = pd.read_excel(path + '/pre_set.xlsx', sheet_name = "min_max",
                          header = None, names = ["symbol"]+[str(i) for i in range(15)]).set_index("symbol").to_dict("split")
        dict_min_max = {temp_dict["index"][i]:temp_dict["data"][i] for i in range(118)}

        #combine all useful properties and save all to a dictionary.
        list_ele = []
        list_symbol = []
        self.dict_ele = {}
        #self.periodic_table=[]
        for k,v in dic_s.items():
            dict_temp = {}
            covalent_radius = float(dic_s[k])
            second_covalent_radius = float(dic_d[k])
            third_covalent_radius = float(dic_t[k])
            X = float(dic_x[k])
            symbol = str(k)
            ele = Element(k)
            #self.periodic_table.append(k)
            
            #min_oxi = int(ele.min_oxidation_state)
            #max_oxi = int(ele.max_oxidation_state)
            min_oxi, max_oxi = min_max(dict_min_max[k])
            oxi_list = valid_OS(dict_min_max[k])
            
            if min_oxi >= 0:
                min_oxi = 0
            if max_oxi <= 0:
                max_oxi = 0
            # oxi_list = [oxi for oxi in range(min_oxi, max_oxi+1)]   ### Change it to the valid OS ###
            
            list_IP = openexcel[k].values.tolist()
            dict_temp = {'symbol':symbol, 'covalent_radius':covalent_radius, 'min_oxi':min_oxi, 'max_oxi':max_oxi, 'oxi_list':oxi_list, 'X':X, 'IP':list_IP, 'second_covalent_radius':second_covalent_radius, 'third_covalent_radius':third_covalent_radius}
            temp_dict = {symbol:dict_temp}
            self.dict_ele.update(temp_dict)
        self.vesta_color = {ele:"rgb"+str((dic_R[ele],dic_G[ele],dic_B[ele])) for ele in list(self.dict_ele.keys())}

        #tune the IP list by the prefer oxidation states
        for ele,os in prefer_OS_ignore_max.items():
            self.dict_ele[ele]["IP"][os] = self.dict_ele[ele]["IP"][os+1]-1

        if not spider:
            self.matrix_of_threshold = np.array(pd.read_csv(path + '/threshold_matrix_looped.csv',header=0, index_col=0))
        else:
            if work_type == "global":
                self.matrix_of_threshold = np.ones([118,118]) * 10
            else:
                self.matrix_of_threshold = np.array(pd.read_csv(path + "/threshold_matrix_looping.csv", header=0, index_col=0))

        local_iter_method = True
        if local_iter_method:
            #print("Applied local charge transfer!")
            self.inorganic_group = {
                                    'V' : {"env":[['O', 'O', 'O', 'O']], "SBO":8, "min":5},#7268
                                    'Cr': {"env":[['O', 'O', 'O', 'O']], "SBO":8, "min":6},
                                    'Mn': {"env":[['O', 'O', 'O', 'O']], "SBO":8, "min":7},#6874
                                    'Fe': {"env":[['O', 'O', 'O', 'O']], "SBO":8, "min":6},#7409
                                    'Mo': {"env":[['O', 'O', 'O', 'O']], "SBO":8, "min":6},#6650
                                    'W' : {"env":[['O', 'O', 'O', 'O']], "SBO":8, "min":6},
                                    'S' : {"env":[['O', 'O', 'O', 'O']], "SBO":8, "min":6},#10266
                                    'Cl': {"env":[['O', 'O', 'O', 'O']], "SBO":8, "min":7},
                                    'Br': {"env":[['O', 'O', 'O', 'O']], "SBO":8, "min":7},
                                    'I' : {"env":[['O', 'O', 'O', 'O']], "SBO":8, "min":7},

                                    'C' : {"env":[['O', 'O', 'O']],      "SBO":4, "min":4},#7667
                                    'N' : {"env":[['O', 'O', 'O']],      "SBO":5, "min":5},#4657
                                    'Si': {"env":[['O', 'O', 'O'],
                                                  ['O', 'O', 'O', 'O']], "SBO":4, "min":4},#35261
                                    'P' : {"env":[['O', 'O', 'O', 'O']], "SBO":5, "min":5},#77818                                   
                                    'Ge': {"env":[['O', 'O', 'O']],      "SBO":4, "min":4},#5395
                                    'As': {"env":[['O', 'O', 'O', 'O']], "SBO":5, "min":5},#4728
                                    'Se': {"env":[['O', 'O', 'O'],
                                                  ['O', 'O', 'O', 'O']], "SBO":6, "min":6},
                                    'Bi': {"env":[['O', 'O', 'O'],
                                                  ['O', 'O', 'O', 'O']], "SBO":5, "min":5},
                                    'B' : {"env":[['O', 'O', 'O'],
                                                  ['O', 'O', 'O', 'O']], "SBO":3, "min":3},#7851
                                    'Al': {"env":[['O', 'O', 'O', 'O']], "SBO":3, "min":3},#8321
                                    } 

                     
        else:
            self.inorganic_group = []
        
        self.Forced_transfer_group = [("B","H")]
          

def CounterSubset(mom,son): 
    mom_counter = Counter(mom)
    son_counter = Counter(son)
    for k,v in son_counter.items():
        if v > mom_counter[k]:
            return False
    return True


def min_max(alist):
    nlist = []
    for i in alist:
        try:
            nlist.append(int(i))
        except:
            None
    nlist.append(0) if nlist == [] else None
    MIN = min(min(nlist),0)
    MAX = max(max(nlist),0)
    return MIN,MAX


def valid_OS(alist):
    nlist = []
    for i in alist:
        try:
            nlist.append(int(i))
        except:
            None
    nlist.append(0) if nlist == [] else None
    return sorted(nlist)
