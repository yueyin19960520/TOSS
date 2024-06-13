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
"""END HERE"""






"""
prefer_OS = {'Sc':3, 'Ti':4, 'V' :5, 'Cr':6, 'Mn':7, 'Fe':3, 'Co':3, 'Ni':2, 'Cu':2, 'Zn':2, 'Ga':3, 'Ge':2, 'As':3, 'Se':4,
             'Y' :3, 'Zr':4, 'Nb':5, 'Mo':4, 'Tc':4, 'Ru':4, 'Rh':3, 'Pd':2, 'Ag':1, 'Cd':2, 'In':3, 'Sn':2, 'Sb':3, 'Te':4,
             'La':3, 'Ce':3, 'Pr':3, 'Nd':3, 'Pm':3, 'Sm':3, 'Eu':3, 'Gd':3, 'Tb':3, 'Dy':3, 'Ho':3, 'Er':3, 'Tm':3, 'Yb':3, 
             'Lu':3, 'Hf':4, 'Ta':5, 'W' :4, 'Re':4, 'Os':4, 'Ir':4, 'Pt':4, 'Au':1, 'Hg':2, 'Tl':1, 'Pb':2, 'Bi':3, 'Po':4,
             'Ac':3, 'Th':4, 'Pa':5, 'U' :4, 'Np':5, 'Pu':4, 'Am':3, 'Cm':3, 'Bk':3, 'Cf':3, 'Es':3, 'Fm':3, 'Md':3, 'No':2, 
             'Lr':3, 'Rf':4, 'Db':5, 'Sg':4, 'Bh':7, 'Hs':8, 'Mt':0, 'Ds':0, 'Rg':0, 'Cn':0}

maxium_OS = {'Sc':3, 'Ti':4, 'V' :5, 'Cr':6, 'Mn':7, 'Fe':7, 'Co':5, 'Ni':4, 'Cu':4, 'Zn':2, 'Ga':3 ,'Ge':4, 'As':5, 'Se':6,
             'Y': 3, 'Zr':4, 'Nb':5, 'Mo':6, 'Tc':7, 'Ru':8, 'Rh':6, 'Pd':4, 'Ag':3, 'Cd':2, 'In':3, 'Sn':4, 'Sb':5, 'Te':6,
             'La':3, 'Ce':4, 'Pr':5, 'Nd':4, 'Pm':3, 'Sm':3, 'Eu':3, 'Gd':3, 'Tb':4, 'Dy':4, 'Ho':3, 'Er':3, 'Tm':3, 'Yb':3, 
             'Lu':3, 'Hf':4, 'Ta':5, 'W' :6, 'Re':7, 'Os':8, 'Ir':9, 'Pt':6, 'Au':5, 'Hg':2, 'Tl':3, 'Pb':4, 'Bi':5, 'Po':6,
             'Ac':3, 'Th':4, 'Pa':5, 'U' :6, 'Np':7, 'Pu':8, 'Am':7, 'Cm':6, 'Bk':5, 'Cf':5, 'Es':4, 'Fm':3, 'Md':3, 'No':3, 
             'Lr':3, 'Rf':4, 'Db':5, 'Sg':6, 'Bh':7, 'Hs':8, 'Mt':0, 'Ds':0, 'Rg':0, 'Cn':0}
"""





"""
#Generate the shell electron configuration
import copy

E_shell = {"1s":[0,0],
           "2s":[0,0],"2p":[0,0,0,0,0,0],
           "3s":[0,0],"3p":[0,0,0,0,0,0],"3d":[0,0,0,0,0,0,0,0,0,0],
           "4s":[0,0],"4p":[0,0,0,0,0,0],"4d":[0,0,0,0,0,0,0,0,0,0],"4f":[0,0,0,0,0,0,0,0,0,0,0,0,0,0],
           "5s":[0,0],"5p":[0,0,0,0,0,0],"5d":[0,0,0,0,0,0,0,0,0,0],"5f":[0,0,0,0,0,0,0,0,0,0,0,0,0,0],
           "6s":[0,0],"6p":[0,0,0,0,0,0],"6d":[0,0,0,0,0,0,0,0,0,0],
           "7s":[0,0],"7p":[0,0,0,0,0,0]}

Band_order = tuple(['1s','2s','2p','3s','3p','4s','3d','4p','5s','4d','5p','6s','4f','5d','6p','7s','5f','6d','7p'])

element_list = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 
                'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga','Ge', 'As', 
                'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 
                'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 
                'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 
                'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 
                'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh',
                'Fl', 'Mc', 'Lv', 'Ts', 'Og']

operations = [
("Cr","3d", "+1"),  ("Cr","4s","-1"), ("Cu","3d", "+1"),  ("Cu","4s","-1"),
("Nb","4d", "+1"),  ("Nb","5s","-1"), ("Mo","4d", "+1"),  ("Mo","5s","-1"),
("Ru","4d", "+1"),  ("Ru","5s","-1"), ("Rh","4d", "+1"),  ("Rh","5s","-1"), 
("Pd","4d", "+1"),  ("Pd","5s","-1"), ("Pd","4d", "+1"),  ("Pd","5s","-1"), 
("Ag","4d", "+1"),  ("Ag","5s","-1"), ("La","4f", "-1"),  ("La","5d","+1"),
("Ce","4f", "-1"),  ("Ce","5d","+1"), ("Gd","4f", "-1"),  ("Gd","5d","+1"),
("Pt","5d", "+1"),  ("Pt","6s","-1"), ("Au","5d", "+1"),  ("Au","6s","-1"),
("Ac","5f", "-1"),  ("Ac","6d","+1"), ("Th","5f", "-1"),  ("Th","6d","+1"),
("Th","5f", "-1"),  ("Th","6d","+1"), ("Pa","5f", "-1"),  ("Pa","6d","+1"),
("U", "5f", "-1"),  ("U" ,"6d","+1"), ("Np","5f", "-1"),  ("Np","6d","+1"),
("Cm","5f", "-1"),  ("Cm","6d","+1"), ("Lr","6d", "-1"),  ("Lr","7p","+1"),
("Ds","6d", "+1"),  ("Ds","7s","-1"), ("Rg","6d", "+1"),  ("Rg","7s","-1")]


def assign_electrons(N, E_shell = E_shell, Band_order = Band_order):
    while N > 0:
        for b in Band_order:
            if set(E_shell[b]) == set([1]):
                continue
            else:
                for i,e in enumerate(E_shell[b]):
                    if e != 1:
                        E_shell[b][i] += 1
                        N -= 1
                        break
            break
    return E_shell


def check(ele, Band_order = Band_order):
    global Electron_Structures
    target = Electron_Structures[ele]
    temp_list = []
    for k in Band_order:
        v = target[k]
        temp_name = ""
        if 1 in v:
            temp_name += k
            temp_name += str(len([i for i in v if i == 1]))
            temp_list.append(temp_name)
    if len(temp_list) > 2:
        out = temp_list[-3] + " " + temp_list[-2] + " " + temp_list[-1]
    elif 2 >= len(temp_list) > 1:
        out = temp_list[-2] + " " + temp_list[-1]
    else:
        out = temp_list[-1]
    return out
            
    
def tune(ele, orbital, direction):
    global Electron_Structures
    if direction == "-1":
        for i,j in enumerate(Electron_Structures[ele][orbital]):
            if j == 0:
                break
        Electron_Structures[ele][orbital][i-1] -= 1
    if direction == "+1":
        for i,j in enumerate(Electron_Structures[ele][orbital]):
            if j == 0:
                break
        Electron_Structures[ele][orbital][i] += 1
    return None
                
            
Electron_Structures = {}
for N,ele in enumerate(element_list):
    E_shell_copy = copy.deepcopy(E_shell)
    Electron_Structures.update({ele:assign_electrons(N+1, E_shell=E_shell_copy)}) 
    
for operation in operations:
    ele = operation[0]
    orbital = operation[1]
    direction = operation[2]
    tune(ele, orbital, direction)
"""