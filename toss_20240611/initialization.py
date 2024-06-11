from digest import get_ele_from_sites
import copy
from pre_set import CounterSubset
import numpy as np
from digest import get_propoteries_of_atom


def get_the_operation_center_atom_idx(i,delta_X,res): #i is the idx of the enumerate center atom.
    #determin the operation center atom
    shell_env_c = res.shell_env_list[i]
    CN = res.shell_CN_list[i]
    X_c = res.dict_ele[get_ele_from_sites(i,res)]["X"]
    
    temp_idx_c = [l for l,cn in enumerate(res.shell_CN_list) if cn == CN]
    temp_idx_c_X_list = [res.dict_ele[get_ele_from_sites(l,res)]["X"] for l in temp_idx_c]
    idx_c = [l for l,x in zip(temp_idx_c, temp_idx_c_X_list) if X_c == x ] 
    
    temp_env_c = [res.shell_env_list[c] for c in idx_c]
    temp_v_c = [res.valence_list[c] for c in idx_c]
    temp_ele_c = [res.elements_list[c] for c in idx_c]
    c_IP = [res.dict_ele[temp_ele_c[p]]['IP'][temp_v_c[p]] for p in range(len(idx_c))]
    temp_c_IP_list = [[l,m,-n] for l,m,n in zip(idx_c, c_IP, temp_env_c)]
    sort_c_IP_list = sorted(temp_c_IP_list, key = (lambda x:[x[1],x[2]]))
    todo_idx_c = sort_c_IP_list[0][0]
    return todo_idx_c


def get_the_operation_neigh_atom_idx(j,res):
    #determine the operation neigh atom
    shell_env_n = res.shell_env_list[j]
    ele_n = get_ele_from_sites(j,res)
    env_ele = []
    for m,n in zip(res.shell_env_list, res.elements_list):
        env_ele.append((m,n))
    idx_n = [i for i,x in enumerate(env_ele) if x == (shell_env_n, ele_n)]
    temp_v_n = [res.valence_list[n] for n in idx_n]  
    temp_ele_n = [res.elements_list[n] for n in idx_n]
    n_X_temp = np.sum([[res.dict_ele[temp_ele_n[i]]['X'] for i in range(len(idx_n))],temp_v_n], axis=0).tolist()
    temp_n_X_list = [[l,m] for l,m in zip(idx_n,n_X_temp)]
    sort_n_X_list = sorted(temp_n_X_list, reverse = True,key = lambda x:x[1])
    todo_idx_n = sort_n_X_list[0][0]
    return todo_idx_n


def local_charge_transfer(i,inorganic_acid_center_idx,res):
    temp_value = res.shell_ele_list[i]
    temp_key = get_ele_from_sites(i,res)
    sbo = sum(res.ori_bo_matrix[i])

    if temp_key in res.inorganic_group:
        if temp_value in res.inorganic_group[temp_key]["env"]:
            if sbo >= res.inorganic_group[temp_key]["SBO"]:
                bond_length_list = [res.matrix_of_length[i][j] for j in res.shell_idx_list[i]]
                bond_length_with_idx = [[l,m] for l,m in zip(res.shell_idx_list[i], bond_length_list)]
                sort_bond_length_with_idx = sorted(bond_length_with_idx, key = lambda x:x[1])
                sort_idx = [sort_bond_length_with_idx[l][0] for l in range(len(sort_bond_length_with_idx))]
                for j in sort_idx:
                    ele_c, r_c, X_c, min_oxi_c, max_oxi_c, v_c = get_propoteries_of_atom(i,res.valence_list, res)
                    ele_n, r_n, X_n, min_oxi_n, max_oxi_n, v_n = get_propoteries_of_atom(j,res.valence_list, res)
                    if max_oxi_c <= v_c or v_n <= min_oxi_n:
                        None
                    else:
                        todo_idx_c = i
                        todo_idx_n = j
                        res.valence_list[todo_idx_c] = v_c + 1
                        res.valence_list[todo_idx_n] = v_n - 1
                        inorganic_acid_flag = True
                inorganic_acid_center_idx.append(i)
                res.min_oxi_list[i] = res.inorganic_group[temp_key]["min"]
    return inorganic_acid_center_idx


def charge_transfer(delta_X,res):
    for i in res.idx:
        idx_list = res.shell_idx_list[i]
        for j in idx_list:
            length_temp = res.matrix_of_length[i][j]
            ele_c, r_c, X_c, min_oxi_c, max_oxi_c, v_c = get_propoteries_of_atom(i,res.valence_list, res)
            ele_n, r_n, X_n, min_oxi_n, max_oxi_n, v_n = get_propoteries_of_atom(j,res.valence_list, res)
            
            if X_c < X_n and abs(X_c - X_n) > delta_X or (ele_c, ele_n) in res.Forced_transfer_group: #becaues X_c < X_n, it has the direction
                todo_idx_c = get_the_operation_center_atom_idx(i,delta_X,res)
                v_c = res.valence_list[todo_idx_c]

                todo_idx_n = get_the_operation_neigh_atom_idx(j,res)
                v_n = res.valence_list[todo_idx_n]
                if max_oxi_c <= v_c or v_n <= min_oxi_n:
                    None
                else:
                    res.valence_list[todo_idx_c] = v_c + 1
                    res.valence_list[todo_idx_n] = v_n - 1
                    res.first_algorithm_flag = True


def get_bond_order(i,j,res):
    bond_length = res.matrix_of_length[i][j]
    r_c_1 = res.dict_ele[get_ele_from_sites(i,res)]["covalent_radius"]
    r_c_2 = res.dict_ele[get_ele_from_sites(i,res)]["second_covalent_radius"]
    r_c_3 = res.dict_ele[get_ele_from_sites(i,res)]["third_covalent_radius"]
    r_n_1 = res.dict_ele[get_ele_from_sites(j,res)]["covalent_radius"]
    r_n_2 = res.dict_ele[get_ele_from_sites(j,res)]["second_covalent_radius"]
    r_n_3 = res.dict_ele[get_ele_from_sites(j,res)]["third_covalent_radius"]
    triple_cut = ((r_c_3 + r_n_3) + (r_c_2 + r_n_2))/200
    if (get_ele_from_sites(i,res), get_ele_from_sites(j,res)) in [("N","O"),("O","N")]:
        double_cut = ((r_c_2 + r_n_2) + (r_c_1 + r_n_1)*2)/300
    else:
        double_cut = ((r_c_2 + r_n_2) + (r_c_1 + r_n_1))/200
    site_i = res.sites[i].species
    site_j = res.sites[j].species

    if bond_length <= triple_cut:
        if (get_ele_from_sites(i,res),get_ele_from_sites(j,res)) in [("C","N"),("N","C")]:
            #print("Triple Bond Accur between %s and %s."%(site_i, site_j))
            temp_str = "Triple Bond Accur between %s and %s."%(site_i, site_j)
            return 3,temp_str

    if bond_length <= double_cut:
        temp_str = "Double Bond Accur between %s and %s."%(site_i, site_j)
        return 2,temp_str
    if double_cut < bond_length:
        temp_str = "Covalent Bond Accur between %s and %s."%(site_i, site_j)
        return 1,temp_str


def change_oxi_limit(delta_X, res):
    original_min_oxi_list = copy.deepcopy(res.min_oxi_list)
    original_max_oxi_list = copy.deepcopy(res.max_oxi_list)

    print_covalent_status = []
    covalent_pair = []
    for i in res.idx:
        ele_c = get_ele_from_sites(i,res)
        X_c = res.dict_ele[ele_c]["X"]
        temp_cov_list = []
        for j in res.shell_idx_list[i]:
            ele_n = get_ele_from_sites(j,res)
            X_n = res.dict_ele[ele_n]["X"]

            if (ele_c, ele_n) not in res.Forced_transfer_group and (ele_n, ele_c) not in res.Forced_transfer_group:
                if ele_n in res.periodic_table.metals and ele_c in res.periodic_table.metals:
                    continue
                else:
                    if abs(X_c - X_n) < delta_X:
                        bond_order,temp_str = get_bond_order(i,j,res)
                        print_covalent_status.append(temp_str)
                        res.max_oxi_list[j] -= bond_order
                        res.min_oxi_list[j] += bond_order
                        temp_cov_list.append(j)
                    else:
                        continue
            else:
                continue
        covalent_pair.append(temp_cov_list)

    #check the reliability: the max cannot less than 0 and the min cannot larger than 0:
    for i in range(len(res.min_oxi_list)):
        if res.min_oxi_list[i] > 0:
            res.min_oxi_list[i] = 0
        if res.max_oxi_list[i] < 0:
            res.max_oxi_list[i] = 0

    print_covalent_status = list(set(print_covalent_status))
    return print_covalent_status, covalent_pair, original_min_oxi_list, original_max_oxi_list


def get_bo_matrix(res):
    bo_matrix = []
    for i in res.idx:
        temp_list = [0 for j in res.idx]
        for j in res.shell_idx_list[i]:
            bo,_ = get_bond_order(i,j,res)
            temp_list[j] += bo
        bo_matrix.append(temp_list)

    #Erase the influence of the atoms share the same local environment but different bond order list
    types_of_atoms = {}
    for i in res.idx:
        key = (get_ele_from_sites(i,res), res.shell_env_list[i], tuple(sorted(res.shell_ele_list[i])))
        if key not in types_of_atoms:
            types_of_atoms[key] = [i]
        else:
            types_of_atoms[key].append(i)

    for k,v in types_of_atoms.items():
        temp = []
        for i in v:
            bo_with_ele = tuple(sorted([(get_ele_from_sites(j,res), bo_matrix[i][j]) for j in res.shell_idx_list[i]]))
            temp.append(bo_with_ele)
        if len(set(temp)) != 1:

            ele_sum = {}
            for i in v:
                temp_dict = {}
                for j in res.shell_idx_list[i]:
                    ele_j = get_ele_from_sites(j,res)
                    if ele_j in temp_dict:
                        temp_dict[ele_j] += bo_matrix[i][j]
                    else:
                        temp_dict[ele_j] = bo_matrix[i][j]
                ele_sum.update({i:temp_dict})

            coor_ele = [[kk for kk,vv in info.items()] for I, info in ele_sum.items()]
            coor_ele = set([n for m in coor_ele for n in m])
            coor_ele_max = {ele: max([n for m in [[info[ele]] for I,info in ele_sum.items()] for n in m]) for ele in coor_ele}

            for i in v:
                for ele in coor_ele:
                    if ele in ele_sum[i]:
                        delta = coor_ele_max[ele] - ele_sum[i][ele] 
                        while delta > 0:
                            op_j = [j for j in res.shell_idx_list[i] if get_ele_from_sites(j,res) == ele]
                            j_bo_with_length = [(j, bo_matrix[i][j], res.matrix_of_length[i][j]) for j in op_j]
                            sort_j_bo_length = sorted(j_bo_with_length, key = lambda x:[x[1], x[2]])
                            j = sort_j_bo_length[0][0]
                            bo_matrix[i][j] += 1
                            bo_matrix[j][i] += 1
                            delta -= 1
                        else:
                            None
                    else:
                        None
        else:
            None
    return bo_matrix



class LOCAL_ALGO():
    def __init__(self, m_id, delta_X, tolerance ,res):

        self.inorganic_acid_flag = True
        self.first_algorithm_flag = True
        #self.inorganic_acid_center_idx = []

    def apply_local_iter_method(self,res):
        while res.inorganic_acid_flag:
            res.inorganic_acid_flag = False
            inorganic_acid_center_idx = []
            for i in res.idx:
                inorganic_acid_center_idx = local_charge_transfer(i,inorganic_acid_center_idx, res)
        self.inorganic_acid_center_idx = list(set(inorganic_acid_center_idx))
        

    def apply_local_algo(self,delta_X,res):
        while res.first_algorithm_flag:
            res.first_algorithm_flag = False
            charge_transfer(delta_X,res)



class INITIAL():
    def __init__(self,tolerance, m_id, delta_X, res):

        self.print_covalent_status, self.covalent_pair, self.original_min_oxi_list, self.original_max_oxi_list = change_oxi_limit(delta_X,res)
        self.bo_matrix = get_bo_matrix(res)
        self.ori_bo_matrix = copy.deepcopy(self.bo_matrix)
        #print(self.bo_matrix)
"""END HERE"""