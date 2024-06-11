import numpy as np
import copy
import math
from digest import get_ele_from_sites
from digest import get_propoteries_of_atom
from initialization import get_bond_order
from post_process import enu_permu_combi
from pre_set import CounterSubset


def get_super_atom_list(res):
    super_atom_idx_list = []
    single_atom_idx_list = []
    for i in res.idx:
        ele_c, r_c, X_c, min_oxi_c, max_oxi_c, v_c = get_propoteries_of_atom(i,res.valence_list, res)
        group_idx = [i]
        group_ele = [ele_c]
        group_X = [X_c]
        group_min_oxi = [min_oxi_c]
        group_max_oxi = [max_oxi_c]
        group_v = [v_c]
        for j in res.shell_idx_list[i]:
            ele_n, r_n, X_n, min_oxi_n, max_oxi_n, v_n = get_propoteries_of_atom(j,res.valence_list, res)
            group_idx.append(j)
            group_ele.append(ele_n)
            group_X.append(X_n)
            group_min_oxi.append(min_oxi_n)
            group_max_oxi.append(max_oxi_n)
            group_v.append(v_n)
        #if group_X[0] == min(group_X) and len(group_X) > 2:
        #if group_X[0] == min(group_X) or len(group_X) == 1:
        if group_X[0] < max(group_X):
            super_atom_idx_list.append(i)
        if len(group_X) == 1:
            single_atom_idx_list.append(i)

    organic_super_atom_idx_list = []
    if res.organic_patch:
        for i in reversed(res.idx):
            if get_ele_from_sites(i,res) == "C":
                organic_super_atom_idx_list.append(i)
                if i in super_atom_idx_list:
                    super_atom_idx_list.remove(i)
            if get_ele_from_sites(i,res) == "H" and res.shell_ele_list[i] == ["C"]:
                super_atom_idx_list.remove(i)
                single_atom_idx_list.append(i)#consider the H link to C is the single atom, because its properties is clear.

    return super_atom_idx_list,single_atom_idx_list, organic_super_atom_idx_list


def local_charge_transfer_for_super_atom(i,j,temp_valence_list, delta_X,check_idx_list,res):  
    ele_c, r_c, X_c, min_oxi_c, max_oxi_c, v_c = get_propoteries_of_atom(i,temp_valence_list,res)
    ele_n, r_n, X_n, min_oxi_n, max_oxi_n, v_n = get_propoteries_of_atom(j,temp_valence_list,res)
    v_c = temp_valence_list[i]
    v_n = temp_valence_list[j]

    if max_oxi_c > v_c and v_n > min_oxi_n and res.bo_matrix[i][j] > 0:
        temp_valence_list[i] = v_c + 1
        temp_valence_list[j] = v_n - 1
        res.bo_matrix[i][j] -= 1
        res.bo_matrix[j][i] -= 1
    else:
        if j in check_idx_list:
            check_idx_list.remove(j)
    return temp_valence_list, check_idx_list


def charge_explosive(i,temp_valence_list,delta_X, res):
    check_idx_list = copy.deepcopy(res.shell_idx_list[i])

    flag_ce = True#ce means charge explosive 
    while flag_ce:
        refine_temp_vl = [temp_valence_list[s] for s in check_idx_list]

        temp_X_list = [res.dict_ele[get_ele_from_sites(s,res)]["X"] for s in check_idx_list]

        temp_X_list_with_idx = [[n,m,l] for n,m,l in zip(check_idx_list, temp_X_list, refine_temp_vl)]

        sort_X_list_with_idx = sorted(temp_X_list_with_idx, reverse = True, key = lambda x:[x[2],x[1]])

        ele_c = get_ele_from_sites(i,res)
        X_c = res.dict_ele[ele_c]["X"]

        sort_idx_list = [l[0] for l in sort_X_list_with_idx if l[1] - X_c > delta_X \
                                                            or (ele_c, get_ele_from_sites(l[0],res)) in res.Forced_transfer_group]

        if len(sort_idx_list) > res.max_oxi_list[i]: #and get_ele_from_sites(i,res) in res.periodic_table.transition_metals:
            record = copy.deepcopy(res.max_oxi_list[i])
            res.max_oxi_list[i] = len(sort_idx_list)

        if len(sort_idx_list) > 0:
            j = sort_idx_list[0]

            temp_valence_list, check_idx_list = local_charge_transfer_for_super_atom(i, j,
                                                                                     temp_valence_list,
                                                                                     delta_X,
                                                                                     check_idx_list, 
                                                                                     res)
            if len(check_idx_list) == 0:
                flag_ce = False
        else:
            flag_ce = False
        #print("103", flag_ce)
    try:
        res.max_oxi_list[i] = record 
    except:
        None

    return temp_valence_list


def new_arbitrary_decision(i, temp_valence_list, delta_X, res): 
    ele_i = get_ele_from_sites(i,res)
    for j in res. shell_idx_list[i]:
        ele_j = get_ele_from_sites(j,res)
        if res.dict_ele[ele_i]["X"] - res.dict_ele[ele_j]["X"] > delta_X and abs(temp_valence_list[i]) <= 4 - res.bo_matrix[i][j]:
            temp_valence_list[i] -= res.bo_matrix[i][j]
            temp_valence_list[j] += res.bo_matrix[i][j]

        elif res.dict_ele[ele_j]["X"] - res.dict_ele[ele_i]["X"] > delta_X <= 4 - res.bo_matrix[i][j]:
            temp_valence_list[j] -= res.bo_matrix[i][j]
            temp_valence_list[i] += res.bo_matrix[i][j]  

        else:
            None
    return temp_valence_list


def reassignment_for_link_atom(link_atom_list_temp, sum_of_valence, res): #maybe it is called "reassignment" better. 
    temp_ele_env_v_list = []
    #for i,l,m,n in zip(res.idx, res.elements_list, res.shell_env_list, sum_of_valence):
    for i,l,m,n in zip(res.idx, res.elements_list, res.shell_ele_list, sum_of_valence):
        if i in link_atom_list_temp:
            temp_ele_env_v_list.append([i,l,m,n])
            
    class_of_link = set([(ilmn[1], tuple(sorted(ilmn[2]))) for ilmn in temp_ele_env_v_list])
    class_of_link_with_idx = []

    for cls in class_of_link:
        temp_cls = []
        for ilmn in temp_ele_env_v_list:
            if (ilmn[1], tuple(sorted(ilmn[2]))) == cls:
                temp_cls.append(ilmn)
        class_of_link_with_idx.append(temp_cls)

    for cls in class_of_link_with_idx:
        valence_of_class = [ilmn[3] for ilmn in cls]
        n_e = sum(valence_of_class)
        n_a = len(cls)
        if n_e % n_a == 0:
            quo = n_e/n_a
            for ilmn in cls:
                sum_of_valence[ilmn[0]] = int(quo)
        else:
            quo = n_e // n_a
            rem = n_e % n_a
            for i, ilmn in enumerate(cls):
                if i < rem:
                    sum_of_valence[ilmn[0]] = int(quo+1)
                else:
                    sum_of_valence[ilmn[0]] = int(quo)
    return sum_of_valence


def reassignment_within_shell(super_atom_idx_list, link_atom_idx_list, sum_of_valence, res):
    for i in super_atom_idx_list:
        op_list_temp = res.shell_idx_list[i]
        op_list = []
        for j in op_list_temp:#make sure all link atoms
            if j in link_atom_idx_list:
                op_list.append(j)
        
        under_saturate = []
        for j in op_list:
            if sum_of_valence[j] > res.min_oxi_list[j]:
                for k in res.idx:
                    if sorted(res.shell_ele_list[k]) == sorted(res.shell_ele_list[j]) and k not in under_saturate and \
                                                get_ele_from_sites(k,res) == get_ele_from_sites(j,res):

                        under_n = sum_of_valence[k] - res.min_oxi_list[k]
                        for t in range(under_n):
                            under_saturate.append(k)  #contains the under_saterated atoms and their all same atom.

        over_saturate = []
        for j in op_list:
            if sum_of_valence[j] < res.min_oxi_list[j]:
                for k in res.idx:
                    if sorted(res.shell_ele_list[k]) == sorted(res.shell_ele_list[j]) and k not in over_saturate and \
                                                get_ele_from_sites(k,res) == get_ele_from_sites(j,res):

                        over_n = res.min_oxi_list[k] - sum_of_valence[k]
                        for t in range(over_n):
                            over_saturate.append(k)    #contains the over_saterated atoms and their all same atom.
        
        if len(under_saturate) > 0 and len(over_saturate) > 0:
            n = min(len(under_saturate), len(over_saturate))
        else:
            n = 0
        
        under_saturate_set = list(set(under_saturate))
        over_saturate_set = list(set(over_saturate))

        while n > 0:

            a_temp_X_list = [res.dict_ele[get_ele_from_sites(j,res)]["X"] for j in under_saturate_set]
            a_X_with_id = [(j,x) for j,x in zip(under_saturate_set, a_temp_X_list)]
            a_sorted_X_with_id = sorted(a_X_with_id, key = lambda x: x[1])
   
            acceptor = a_sorted_X_with_id[0][0]

            d_temp_X_list = [res.dict_ele[get_ele_from_sites(j,res)]["X"] for j in over_saturate_set]
            d_X_with_id = [(j,x) for j,x in zip(over_saturate_set, d_temp_X_list)]
            d_sorted_X_with_id = sorted(d_X_with_id, reverse = True, key = lambda x: x[1])

            doner = d_sorted_X_with_id[0][0]

            sum_of_valence[acceptor] -= 1
            sum_of_valence[doner] += 1
            
            if sum_of_valence[acceptor] == res.min_oxi_list[acceptor]:
                under_saturate_set.remove(acceptor)
            if sum_of_valence[doner] == res.min_oxi_list[doner]:
                over_saturate_set.remove(doner)

            n -= 1
            #print("226", n)

    return sum_of_valence


def set_valence_of_link_atoms(link_atom_idx_list, sum_of_valence, arti_n, res):
    check_diff, n = [], 0
    n += arti_n
    
    for l in link_atom_idx_list:
        now_v = sum_of_valence[l]
        now_valid_os = res.dict_ele[get_ele_from_sites(l, res)]["oxi_list"]
        if now_v not in now_valid_os:
            min_valid_os = min([os for os in now_valid_os if os > now_v])
            n += min_valid_os - now_v
            sum_of_valence[l] = min_valid_os

    return n, sum_of_valence


def set_valence_of_super_atoms(super_atom_idx_list, sum_of_valence, n, res):
    for l in super_atom_idx_list:
        now_v = sum_of_valence[l]
        now_valid_os = res.dict_ele[get_ele_from_sites(l, res)]["oxi_list"]
        if now_v not in now_valid_os:
            max_valid_os = max([os for os in now_valid_os if os < now_v])
            n += max_valid_os - now_v
            sum_of_valence[l] = max_valid_os
    return n, sum_of_valence


def validate_solutions(solution_candidate, val, res, over_oxided_switch=False):
    solutions = []
    for solution in solution_candidate:
        flags = []
        for n_charge_back, (class_key, valence_now_for_class) in zip(solution, val.items()):
            if over_oxided_switch:
                valence_later_for_class = valence_now_for_class + n_charge_back
            else:
                valence_later_for_class = valence_now_for_class - n_charge_back
                
            valid_os_for_class = res.dict_ele[class_key[0]]["oxi_list"]
            flags.append(True) if valence_later_for_class in valid_os_for_class else flags.append(False)
        if all([flags]):
            solutions.append(solution)
    return solutions


def enumerate_path(n, super_atom_idx_list, sum_of_valence, res):
    #print("253", sum_of_valence)
    super_atom_ele_list = [get_ele_from_sites(e,res) for e in super_atom_idx_list]
    super_atom_env_list = [tuple(sorted(res.shell_ele_list[e])) for e in super_atom_idx_list]

    super_atom_env_with_ele_list = [(i,j) for i, j in zip(super_atom_ele_list, super_atom_env_list)]
    super_atom_env_with_ele_uni_list = list(set(super_atom_env_with_ele_list))

    num = [0 for i in range(len(super_atom_env_with_ele_uni_list))]
    for i,e in enumerate(super_atom_env_with_ele_uni_list):
        for j in super_atom_env_with_ele_list:
            if e == j:
                num[i] = num[i] + 1

    ex_idx = {}
    for e in super_atom_env_with_ele_uni_list:
        for i,ee in zip(super_atom_idx_list,super_atom_env_with_ele_list):
            if e == ee:
                ex_idx[ee] = i
                break

    val = {}
    for e in super_atom_env_with_ele_uni_list:
        for i,ee in zip(super_atom_idx_list,super_atom_env_with_ele_list):
            if e == ee:
                val[ee] = sum_of_valence[i]
                break

    uni_with_num = [(i,j) for i,j in zip(super_atom_env_with_ele_uni_list, num)]
    num = [i[1] for i in uni_with_num if n//i[1] != 0]             #erase the super atoms whose number less than n!!!!!  
    super_atom_env_with_ele_uni_list = [i[0] for i in uni_with_num if n//i[1] != 0]  #same upper one!!!!
    mathmetical_limit = [n//i for i in num]

    upper_oxi_limit = [val[i] for i in super_atom_env_with_ele_uni_list]

    lower_oxi_limit = [max(res.min_oxi_list[ex_idx[i]], 0) for i in super_atom_env_with_ele_uni_list]

    oxi_limit = [u-l for u,l in zip(upper_oxi_limit, lower_oxi_limit)]

    limit = [min(oxi_limit[i], mathmetical_limit[i]) for i in range(len(super_atom_env_with_ele_uni_list))]

    mtl = 1
    for lmt in limit:
        mtl = mtl * (lmt+1)
    if mtl > 2000000:
        raise ValueError("The number of combination is too large!")

    solution_candidate = enu_permu_combi(limit)

    solution_candidate = [s for s in solution_candidate if np.dot(s, num) == n]
    
    solution = validate_solutions(solution_candidate, val, res)

    PATH_temp = []
    for i in range(len(solution)):
        path = []
        for s,p in zip(solution[i], super_atom_env_with_ele_uni_list):
            ele_here = p[0]
            cur_v = val[p]
            vp = 1
            for t in range(s):
                path.append([res.dict_ele[ele_here]["IP"][cur_v-vp], p[1]])
                vp = vp+1 
        PATH_temp.append(path)      

    PATH = []
    for path in PATH_temp:
        sorted_path = sorted(path, reverse = True, key = (lambda x:[x[0],x[1]]))
        PATH.append(sorted_path)

    return PATH


def get_the_operation_path_by_energy(PATH, super_atom_idx_list, sum_of_valence, res):
    if len(PATH) == 1:
        PATH_idx = 0
    else:
        Energy_list = []
        for path_idx, path in enumerate(PATH):
            E = 0
            E_prime = 0
            r_valence_list = copy.deepcopy(sum_of_valence)
            for way in path:
                r_super_atom_ip = [res.dict_ele[get_ele_from_sites(i,res)]["IP"][r_valence_list[i]-1] for i in super_atom_idx_list]
                r_super_atom_env = [tuple(sorted(res.shell_ele_list[i])) for i in super_atom_idx_list]
                r_idx_ip_env = [(i,j,k) for i,j,k in zip(super_atom_idx_list, r_super_atom_ip,r_super_atom_env)]

                for iie in r_idx_ip_env:
                    if iie[1] == way[0] and iie[2] == way[1]:
                        E += way[0]
                        E_prime -= sum([res.dict_ele[ele]["X"] for ele in way[1]])
                        r_valence_list[iie[0]] -= 1

            Energy_list.append((path_idx, round(E,2), round(E_prime,2), -len(path)))
            
        sorted_Energy_list = sorted(Energy_list, key=lambda x:[x[1], x[2],x[3]])

        PATH_idx = sorted_Energy_list[-1][0]

    return PATH_idx


def group_charge_transfer_by_path(PATH,n,super_atom_idx_list,sum_of_valence,res):
    super_atom_IP_list = [res.dict_ele[get_ele_from_sites(e,res)]["IP"][sum_of_valence[e]-1] for e in super_atom_idx_list]

    super_atom_env_list = [tuple(sorted(res.shell_ele_list[e])) for e in super_atom_idx_list]

    super_atom_IP_with_env_and_idx = [[l,m,n] for l,m,n in zip(super_atom_idx_list, 
                                                               super_atom_IP_list,
                                                               super_atom_env_list)]

    way = PATH.pop(0)
    ip = way[0]
    env = way[1]
    operation_list = [l for [l,m,n] in super_atom_IP_with_env_and_idx if m==ip and n==env]
    if len(operation_list) <= n:
        for i in operation_list:
            sum_of_valence[i] = sum_of_valence[i] -1
            n = n - 1
    return n, sum_of_valence


def flatten_link_atom(link_atom_idx_list,sum_of_valence, res):
    temp_env_list = [tuple(sorted(res.shell_ele_list[i])) for i in link_atom_idx_list]
    temp_ele_list = [res.elements_list[i] for i in link_atom_idx_list]
    temp_ele_with_env = [(i,j) for i,j in zip(temp_ele_list,temp_env_list)]
    class_here = set(temp_ele_with_env)
    temp_ele_with_env_with_idx = [(i,j) for i,j in zip(link_atom_idx_list, temp_ele_with_env)]
    arti_n = 0

    for clss in class_here:
        valence_check = []
        op_list = []

        for i in temp_ele_with_env_with_idx:
            if clss == i[1]:
                valence_check.append(sum_of_valence[i[0]])
                op_list.append(i[0])

        if len(set(valence_check)) != 1:
            max_os = max(valence_check)
            arti_n += sum([max_os - valence_check[l] for l in range(len(valence_check))])

            for i in op_list:
                sum_of_valence[i] = max_os

    return arti_n


def over_oxided(link_atom_idx_list, sum_of_valence, n, res):
    N = abs(n)

    species_list = [(res.elements_list[i],tuple(sorted(res.shell_ele_list[i])),sum_of_valence[i]) for i in link_atom_idx_list]
    species_set = list(set(species_list))
    species_dict = {(i[0],i[1]):[] for i in species_set}
    species_now_valence_dict = {(i[0],i[1]):i[2] for i in species_set}

    for j in link_atom_idx_list:
        species_dict[(res.elements_list[j], tuple(sorted(res.shell_ele_list[j])))].append(j)

    num_limit = [species_list.count(i) for i in species_set]

    mathmetical_limit = [N//i for i in num_limit]

    fake_link_atom_list = [j for j in link_atom_idx_list if CounterSubset(list(set(link_atom_idx_list)), 
                                                                          list(set(res.shell_idx_list[j])))]

    record = {get_ele_from_sites(i,res):res.dict_ele[get_ele_from_sites(i,res)]["X"] for i in fake_link_atom_list}
    max_X = max([res.dict_ele[get_ele_from_sites(i,res)]["X"] for i in link_atom_idx_list])
    for i in fake_link_atom_list:
        res.dict_ele[get_ele_from_sites(i,res)]["X"] = max_X+0.01 

    top_os_limit = [res.max_oxi_list[v[0]] if v[0] in fake_link_atom_list else -1 for k,v in species_dict.items()]

    #limit = [i if j[2] == 0 else min(i, -1-j[2]) for i,j in zip(mathmetical_limit, species_set)]
    #limit = [i if j[2] == 0 else min(i, 0-j[2]) for i,j in zip(mathmetical_limit, species_set)]
    #limit = [top if j[2] == 0 else min(i, top-j[2]) for top,i,j in zip(top_os_limit, mathmetical_limit, species_set)]
    limit = [min(i, top-j[2]) for top,i,j in zip(top_os_limit, mathmetical_limit, species_set)]


    mtl = 1
    for lmt in limit:
        mtl *= (lmt + 1)
    if mtl > 2000000:
        raise ValueError("The number of combination is too large!")

    solution_candidate = enu_permu_combi(limit)

    solution_candidate = [s for s in solution_candidate if np.dot(s, num_limit) == N]
    
    solutions = validate_solutions(solution_candidate, species_now_valence_dict, res, over_oxided_switch=True)

    energy_per_solution = []

    for solution in solutions:
        sum_X = 0
        effective_val = 0
        temp_sum_of_valence = copy.deepcopy(sum_of_valence)
        for i,e in enumerate(solution):
            operation_list = species_dict[(species_set[i][0], species_set[i][1])]
            while e != 0:
                for j in operation_list:
                    effective_val += temp_sum_of_valence[j]
                    temp_sum_of_valence[j] += 1
                    sum_X += res.dict_ele[res.elements_list[j]]["X"]
                e -= 1
                #print("460", e)
        energy_per_solution.append((round(sum_X,2), temp_sum_of_valence, effective_val))

    sum_of_valence = sorted(energy_per_solution, key = lambda x:[x[0],x[2]])[0][1]

    for i in fake_link_atom_list:
        res.dict_ele[get_ele_from_sites(i,res)]["X"] = record[get_ele_from_sites(i,res)] 

    return sum_of_valence


def operate_bo_matrix(super_atom_idx_list,organic_super_atom_idx_list, res):#only apply this in the organic structures.
    for i in super_atom_idx_list:
        if len(res.shell_ele_list[i]) > 1:
            for j in res.shell_idx_list[i]:
                if j in super_atom_idx_list:
                    res.bo_matrix[i][j] = 0
                #if j in organic_super_atom_idx_list:
                    #res.bo_matrix[i][j] = 0
                    #res.bo_matrix[j][i] = 0
        else:
            None #like H only link to one atom, it must lose one electron to that atom.
    
    octet_dict = {"S":{"top":6, "bottom":2}, 
                  "C":{"top":4, "bottom":4}, 
                  "N":{"top":5, "bottom":3}, 
                  "P":{"top":5, "bottom":3}} #super_atom_idx_list, single_atom_idx, organic_super_atom_idx_list

    organic_elements = list(octet_dict.keys()) + ["O","H"]
    sum_bo_list = []
    for i in res.idx:
        temp_sum_bo = sum(res.bo_matrix[i])
        for j in res.shell_idx_list[i]:
            if get_ele_from_sites(j,res) not in organic_elements: #octet rules + HO
                temp_sum_bo -= 1
        sum_bo_list.append(temp_sum_bo)

    for i in res.idx:
        abs_sum_of_bo = 0
        for j in res.shell_idx_list[i]:
            bo,_ = get_bond_order(i,j,res)

            if res.dict_ele[get_ele_from_sites(j,res)]["X"] >= res.dict_ele[get_ele_from_sites(i,res)]["X"]:
                abs_sum_of_bo += bo
            else:
                abs_sum_of_bo -= bo
        SIGN = "top" if abs_sum_of_bo >= 0 else "bottom"
            
        ele_i = get_ele_from_sites(i,res)
        if ele_i in list(octet_dict.keys()) and sum_bo_list[i] > octet_dict[ele_i][SIGN]:
            organic_shell_idx_list = [j for j in res.shell_idx_list[i] if get_ele_from_sites(j,res) in organic_elements] #octet rules + HO
            sum_bo_check = [(get_ele_from_sites(j,res), res.bo_matrix[i][j]) for j in organic_shell_idx_list]
            
            if sum_bo_check.count(("O",2)) > 1:
                idx_with_env_ele_bo = [(j, 
                                        res.shell_env_list[j], 
                                        get_ele_from_sites(j,res), 
                                        res.bo_matrix[i][j]) for j in organic_shell_idx_list]

                O_info = [info for info in idx_with_env_ele_bo if info[2] == "O" and info[3] == 2]
                sort_O_info = sorted(O_info, reverse = True, key = lambda x:x[1]) 
                j = sort_O_info[0][0]
                res.bo_matrix[i][j] -= 1
                res.bo_matrix[j][i] -= 1
                sum_bo_list[i] -= 1
                sum_bo_list[j] -= 1

            elif sum_bo_check.count(("O",2)) == 1 and sum_bo_check.count(("C",2)) == 1:
                idx_with_env_ele_bo = [(j, 
                                        res.shell_env_list[j], 
                                        get_ele_from_sites(j,res), 
                                        res.bo_matrix[i][j]) for j in organic_shell_idx_list]

                C_idx = [info for info in idx_with_env_ele_bo if info[2] == "C" and info[3] == 2][0][0]
                O_idx = [info for info in idx_with_env_ele_bo if info[2] == "O" and info[3] == 2][0][0]
                if sum_bo_list[C_idx] > octet_dict[ele_i]["top"]:
                    j = C_idx
                else:
                    j = O_idx
                res.bo_matrix[i][j] -= 1
                res.bo_matrix[j][i] -= 1
                sum_bo_list[i] -= 1
                sum_bo_list[j] -= 1

            else:
                if get_ele_from_sites(i,res) != "C":
                    for j in organic_shell_idx_list:
                            ele_j = get_ele_from_sites(j,res)
                            if ele_j in list(octet_dict.keys()) and sum_bo_list[j] > octet_dict[ele_j][SIGN]:
                                if res.bo_matrix[i][j] == 2:
                                    sum_bo_list[i] -= 1
                                    sum_bo_list[j] -= 1
                                    res.bo_matrix[i][j] -= 1
                                    res.bo_matrix[j][i] -= 1
                                    break

        if ele_i in list(octet_dict.keys()) and sum_bo_list[i] < octet_dict[ele_i][SIGN]:
            organic_shell_idx_list = [j for j in res.shell_idx_list[i] if get_ele_from_sites(j,res) in organic_elements]
            for j in organic_shell_idx_list:
                if get_ele_from_sites(j,res) == "C":
                    if res.bo_matrix[i][j] == 1:
                        sum_bo_list[i] += 1
                        sum_bo_list[j] += 1
                        res.bo_matrix[i][j] += 1
                        res.bo_matrix[j][i] += 1
                        break                  
    return None


def retrive_oxi(res, super_atom_idx_list, link_atom_idx_list):
    for j in link_atom_idx_list:
        if res.min_oxi_list[j] == 0:
            #the ability of recepting electrons should be retrived!
            #retrived = len(set(res.covalent_pair[j]))
            retrived = res.original_min_oxi_list[j]
            res.min_oxi_list[j] = retrived

    for i in super_atom_idx_list:
        if res.max_oxi_list[i] == 0:
            res.max_oxi_list[i] = res.dict_ele[get_ele_from_sites(i,res)]["max_oxi"]
    return None


def break_bond(res,i,j):
    shell_idx_list[i].remove(j)
    shell_CN_list[i] = len(shell_idx_list[i])
    shell_ele_list[i] = [get_ele_from_sites(k,res) for k in shell_idx_list[i]]
    shell_X_list[i] = [res.dict_ele[get_ele_from_sites(k,res)]["X"] for k in shell_idx_list[i]]
    shell_env_list[i] = round(sum(shell_idx_list[i]),2)


def link_bond(res,ori_res,i):
    res.shell_idx_list[i] = copy.deepcopy(ori_res.shell_idx_list[i])
    res.shell_CN_list[i] = copy.deepcopy(ori_res.shell_CN_list[i])
    res.shell_ele_list[i] = copy.deepcopy(ori_res.shell_ele_list[i])
    res.shell_X_list[i] = copy.deepcopy(ori_res.shell_X_list[i])
    res.shell_env_list[i] = copy.deepcopy(ori_res.shell_env_list[i])


def cation_bond(res,sum_of_valence):
    waiting_for_break = []
    for i in res.idx:
        for j in res.shell_idx_list[i]:
            if sum_of_valence[i] > 0 and sum_of_valence[j] > 0:
                if get_ele_from_sites(i,res) in res.periodic_table.metals and get_ele_from_sites(j,res) in res.periodic_table.metals:
                    waiting_for_break.append((i,j))
                else:
                    None
            else:
                None

    #the fake ZERO valence!
    for i in res.idx:
        for j in res.shell_idx_list[i]:
            a = sum_of_valence[i]
            b = sum_of_valence[j]
            Xa = res.dict_ele[get_ele_from_sites(i,res)]["X"]
            Xb = res.dict_ele[get_ele_from_sites(j,res)]["X"]
            if a + b > 0 and a * b == 0:
                if abs(Xa - Xb) > 0.1:
                    if get_ele_from_sites(i,res) in res.periodic_table.metals and get_ele_from_sites(j,res) in res.periodic_table.metals:
                        waiting_for_break.append((i,j))
                    else:
                        None
                else:
                    None
            else:
                    None

    Cation_bond = True if waiting_for_break != [] else False

    #for pair in waiting_for_break:
        #break_bond(res, pair[0], pair[1])

    #if [] in res.shell_idx_list:
        #raise ValueError("Superatom is isolated!")

    #res.bo_matrix = copy.deepcopy(res.ori_bo_matrix)

    return Cation_bond, waiting_for_break


def process_cation_bond(break_pair_list, res):
    if break_pair_list != []:

        for pair in break_pair_list:
            i = pair[0]
            j = pair[1]
            
            res.shell_idx_list[i].remove(j)

        break_flag = False
        repair = []
        for i, idx_list in enumerate(res.shell_idx_list):
            if idx_list == []:
                repair += [pair for pair in break_pair_list if pair[0] == i]
                break_flag = True
            else:
                None
        for pair in repair:
            i = pair[0]
            j = pair[1]
            res.shell_idx_list[i].append(j)
            res.shell_idx_list[j].append(i)

        for i in res.idx:
            res.shell_ele_list[i] = [get_ele_from_sites(j,res) for j in res.shell_idx_list[i]]
            res.shell_CN_list[i] = len(res.shell_ele_list[i])
            res.shell_X_list[i] = [res.dict_ele[ele]["X"] for ele in res.shell_ele_list[j]]
            res.shell_env_list[i] = round(sum(res.shell_X_list[i]),2)
    else:
        break_flag = False

    res.bo_matrix = copy.deepcopy(res.ori_bo_matrix)
    return break_flag


def lower_oxidation_level(res, sum_of_valence, link_atom_idx_list, n, lower_recording):
    sl_pair_list = []
    operating_link_atom_list = copy.deepcopy(link_atom_idx_list)

    for j in reversed(operating_link_atom_list): # Backword enumerate link atom list, why?
        if sum_of_valence[j] == -1:
            operating_link_atom_list.remove(j) # DO NOT consider the -1 valence link atom, which answer the question backward enumerating.
    
    for j in operating_link_atom_list: # Shorter list
        for i in res.shell_idx_list[j]:
            ele_j = get_ele_from_sites(j,res)
            ele_i = get_ele_from_sites(i,res)
            sl_pair_list.append((j, ele_i, ele_j, round(abs(res.dict_ele[ele_j]["X"] - res.dict_ele[ele_i]["X"]),2), lower_recording[j]["level"]))
            # sl_pait_list[idx] = (index_j, ele_i, ele_j, diff_X, lower_level)

    sort_sl_pair_list = sorted(list(sl_pair_list), key = lambda x:(x[4],x[3])) # sort the list by diff_X and lover_level

    min_delta_x = sort_sl_pair_list[0][3] # Minimum diff_X

    op_link_atom_set = set([p[0] for p in sort_sl_pair_list if p[3] == min_delta_x]) # only operate the minimum diff_X pair

    for j in op_link_atom_set:
        lower_recording[j]["level"] += 1
        res.min_oxi_list[j] += 1
        lower_recording[j]["min"] += 1
        sum_of_valence[j] += 1
        n += 1

    return n, sum_of_valence, lower_recording


def jump_to_resonance(res, sum_of_valence, super_atom_idx_list, n):
    alkali = res.periodic_table.alkali
    earth_alkali = res.periodic_table.earth_alkali

    # Combine alkali and earth alkali elements into one dictionary
    set_value_ele_dict = {**alkali, **earth_alkali}
    
    # Find resonance atoms in super_atom_idx_list
    resonance_atoms = [i for i in super_atom_idx_list if get_ele_from_sites(i, res) in set_value_ele_dict]
    
    if resonance_atoms:
        while n > 0:
            # Sort super_atom_idx_list by sum_of_valence in descending order
            sorted_idx_val = sorted(super_atom_idx_list, key=lambda i: sum_of_valence[i], reverse=True)
            op_i = sorted_idx_val[0]  # Get the index with the highest valence sum
            
            ele_i = get_ele_from_sites(op_i, res)
            valid_os_ele_i = res.dict_ele[ele_i]["oxi_list"]
            
            # Find the maximum valid oxidation state less than current valence sum
            v = max(os for os in valid_os_ele_i if os < sum_of_valence[op_i])
            
            # Decrease n by the difference and update sum_of_valence
            n -= (sum_of_valence[op_i] - v)
            sum_of_valence[op_i] = v
        
        return True
    return False



class POLYHEDRON_ALGO():
    def __init__(self, delta_X, res):
        #res.alloy_flag = False
        if res.alloy_flag:
            n, sum_of_valence, super_atom_idx_list, link_atom_idx_list, single_atom_idx_list = 0,[0 for o in res.idx],[],[],[]
            self.ori_n,self.ori_super_atom_idx_list,self.ori_sum_of_valence = 0, [], []
        else:
            Cation_Bond = True
            waiting_for_break = []

            while Cation_Bond:
                break_flag = process_cation_bond(waiting_for_break, res)

                if break_flag:
                    break
                else:
                    None

                sum_of_valence,\
                super_atom_idx_list,\
                single_atom_idx_list,\
                organic_super_atom_idx_list,\
                link_atom_idx_list = self.main_part(res, delta_X)

                Cation_Bond, waiting_for_break = cation_bond(res, sum_of_valence)

        self.super_atom_idx_list = super_atom_idx_list
        self.single_atom_idx_list = single_atom_idx_list
        self.link_atom_idx_list = link_atom_idx_list
        self.sum_of_valence = sum_of_valence
        

    def main_part(self, res, delta_X):
        sum_of_valence = [0 for l in res.idx]
        super_atom_idx_list,\
        single_atom_idx_list,\
        organic_super_atom_idx_list = get_super_atom_list(res)

        link_atom_idx_list = list(set(res.idx).difference(set(super_atom_idx_list)))
        retrive_oxi(res, super_atom_idx_list, link_atom_idx_list)

        if res.organic_patch:
            operate_bo_matrix(super_atom_idx_list,organic_super_atom_idx_list, res)

        for i in super_atom_idx_list:
            temp_valence_list = [0 for l in res.idx]
            temp_valence_list = charge_explosive(i, temp_valence_list, delta_X, res)
            sum_of_valence = np.sum([sum_of_valence, temp_valence_list], axis = 0).tolist()

        if res.organic_patch:
            for i in organic_super_atom_idx_list:
                temp_valence_list = [0 for l in res.idx]
                temp_valence_list = new_arbitrary_decision(i, temp_valence_list, delta_X, res)
                sum_of_valence = np.sum([sum_of_valence, temp_valence_list], axis = 0).tolist()
            # new adding
            for i in reversed(organic_super_atom_idx_list):
                if sum_of_valence[i] < res.min_oxi_list[i]:
                    organic_super_atom_idx_list.remove(i)
                if sum_of_valence[i] > res.max_oxi_list[i]:
                    super_atom_idx_list.append(i)
                    
        for i in reversed(super_atom_idx_list):
            if sum_of_valence[i] < 0:
                super_atom_idx_list.remove(i)

        link_atom_idx_list = list(set(res.idx).difference(set(super_atom_idx_list)))
        link_atom_idx_list = list(set(link_atom_idx_list).difference(set(single_atom_idx_list)))
        link_atom_idx_list = list(set(link_atom_idx_list).difference(set(organic_super_atom_idx_list)))

        sum_of_valence = reassignment_within_shell(super_atom_idx_list,
                                                   link_atom_idx_list,
                                                   sum_of_valence, 
                                                   res)

        sum_of_valence = reassignment_within_shell(organic_super_atom_idx_list,
                                                   link_atom_idx_list,
                                                   sum_of_valence, 
                                                   res)

        sum_of_valence = reassignment_for_link_atom(link_atom_idx_list,
                                                    sum_of_valence, 
                                                    res)

        arti_n  = flatten_link_atom(link_atom_idx_list,sum_of_valence, res)

        n, sum_of_valence = set_valence_of_link_atoms(link_atom_idx_list, sum_of_valence, arti_n, res) 

        n, sum_of_valence = set_valence_of_super_atoms(super_atom_idx_list, sum_of_valence, n, res) 

        for i in res.idx:
            if get_ele_from_sites(i,res) == "H" and sum_of_valence[i] == 1:
                if i in reversed(super_atom_idx_list):
                    super_atom_idx_list.remove(i)

        #super_atom_idx_list = list(set(super_atom_idx_list).difference(set(res.inorganic_acid_center_idx)))

        self.ori_n = copy.deepcopy(n)
        self.ori_super_atom_idx_list = copy.deepcopy(super_atom_idx_list)
        self.ori_sum_of_valence = copy.deepcopy(sum_of_valence)

        if n > 0:
            if len(super_atom_idx_list) == 0:
                raise KeyError("There is NO Super Atom anymore!")

            PATH = enumerate_path(n,super_atom_idx_list,sum_of_valence,res)

            if len(PATH) == 0:
                jump_flag = jump_to_resonance(res,sum_of_valence,super_atom_idx_list, n)
                if jump_flag:
                    return sum_of_valence, super_atom_idx_list, single_atom_idx_list, organic_super_atom_idx_list, link_atom_idx_list
                else:
                    None

            lower_recording = {j:{"level":0, "min":res.min_oxi_list[j]} for j in link_atom_idx_list}

            #while len(PATH) == 0 and [sum_of_valence[j] for j in link_atom_idx_list].count(-1) != len(link_atom_idx_list):
            all_lg_minus1 = list(filter(lambda x:x>=-1, [sum_of_valence[j] for j in link_atom_idx_list]))

            while len(PATH) == 0 and len(all_lg_minus1) < len(link_atom_idx_list):
                # FIND a path or ALL link atoms' valences change to -1, stop the while!

                n, sum_of_valence, lower_recording = lower_oxidation_level(res, 
                                                                          sum_of_valence, 
                                                                          link_atom_idx_list,
                                                                          n, 
                                                                          lower_recording)

                PATH = enumerate_path(n,super_atom_idx_list,sum_of_valence,res)

                all_lg_minus1 = list(filter(lambda x:x>=-1, [sum_of_valence[j] for j in link_atom_idx_list]))

            if len(PATH) ==0:
                raise ValueError("There is no porper way for putting charge back!")
            else:
                None #FIND the proper way!
                

            lowest_energy_PATH_idx = get_the_operation_path_by_energy(PATH, super_atom_idx_list, sum_of_valence, res)

            while n > 0:
                n, sum_of_valence = group_charge_transfer_by_path(PATH[lowest_energy_PATH_idx],
                                                                  n,
                                                                  super_atom_idx_list,
                                                                  sum_of_valence,
                                                                  res)
                #print("872",n)

        if n < 0:
            sum_of_valence = over_oxided(link_atom_idx_list, sum_of_valence, n, res)

        return sum_of_valence, super_atom_idx_list, single_atom_idx_list, organic_super_atom_idx_list, link_atom_idx_list

    def uniformity(self,vl,res, atom_list):
        ele_list = [get_ele_from_sites(i,res) for i in atom_list]
        env_list = [tuple(sorted(res.shell_ele_list[i])) for i in atom_list]
        val_list = [vl[i] for i in atom_list]
        species_list = [(i,j,k) for i,j,k in zip(ele_list, env_list, val_list)]
        species_uni_list = list(set(species_list))
        return species_uni_list
"""END HERE"""