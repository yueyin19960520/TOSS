from digest import get_ele_from_sites
import copy

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


class INITIAL():
    def __init__(self,tolerance, m_id, delta_X, res):

        self.print_covalent_status, self.covalent_pair, self.original_min_oxi_list, self.original_max_oxi_list = change_oxi_limit(delta_X,res)
        self.bo_matrix = get_bo_matrix(res)
        self.ori_bo_matrix = copy.deepcopy(self.bo_matrix)
        #print(self.bo_matrix)
"""END HERE"""