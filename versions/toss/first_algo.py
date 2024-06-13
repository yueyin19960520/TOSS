from pre_set import CounterSubset
import numpy as np
from digest import get_ele_from_sites
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


class FIRST_ALGO():
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
		

	def apply_first_algo(self,delta_X,res):
		while res.first_algorithm_flag:
			res.first_algorithm_flag = False
			charge_transfer(delta_X,res)
"""END HERE"""

"""
def local_charge_transfer_old(i,inorganic_acid_center_idx,res):
    temp_value = res.shell_ele_list[i]
    temp_key = get_ele_from_sites(i,res)

    for group_dict in res.inorganic_group:
        if temp_key == list(group_dict.keys())[0]:
            if CounterSubset(temp_value,group_dict[temp_key]):     
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
    return inorganic_acid_center_idx


def local_charge_transfer_new(i,inorganic_acid_center_idx,res):
    temp_value = res.shell_ele_list[i]
    temp_key = get_ele_from_sites(i,res)
    inner_flag = True
    for group_dict in res.inorganic_group:
        if temp_key == list(group_dict.keys())[0]:
            if CounterSubset(temp_value,group_dict[temp_key]):    
                bond_length_list = [res.matrix_of_length[i][j] for j in res.shell_idx_list[i]]
                bond_length_with_idx = [[l,m] for l,m in zip(res.shell_idx_list[i], bond_length_list)]
                sort_bond_length_with_idx = sorted(bond_length_with_idx, key = lambda x:x[1])
                sort_idx = [sort_bond_length_with_idx[l][0] for l in range(len(sort_bond_length_with_idx))]
                while inner_flag:
                    inner_flag = False
                    for j in sort_idx:
                        ele_c, r_c, X_c, min_oxi_c, max_oxi_c, v_c = get_propoteries_of_atom(i,res.valence_list, res)
                        ele_n, r_n, X_n, min_oxi_n, max_oxi_n, v_n = get_propoteries_of_atom(j,res.valence_list, res)

                        if max_oxi_c > v_c and v_n > min_oxi_n and res.bo_matrix[i][j] > 0:
                            inner_flag = True
                            todo_idx_c = i
                            todo_idx_n = j
                            res.valence_list[todo_idx_c] = v_c + 1
                            res.valence_list[todo_idx_n] = v_n - 1
                            inorganic_acid_flag = True
                inorganic_acid_center_idx.append(i)
    return inorganic_acid_center_idx
 
"""