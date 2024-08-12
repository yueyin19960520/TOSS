from digest import get_ele_from_sites
from result import RESULT
from multiprocessing.dummy import Pool as ThreadPool
import copy
import math
import numpy as np


def spider_pair_length_with_CN_normed(res):
    temp_pair_info = {}
    for i in res.idx:
        length_list = res.matrix_of_length[i]
        for j in res.shell_idx_list[i]:
            ele_c = get_ele_from_sites(i,res)
            ele_n = get_ele_from_sites(j,res)
            v_c = str(res.sum_of_valence[i])
            v_n = str(res.sum_of_valence[j])
            CN_c = len(res.shell_ele_list[i])
            CN_n = len(res.shell_ele_list[j])
            SCN = CN_c + CN_n

            if res.periodic_table.elements_list.index(ele_c) < res.periodic_table.elements_list.index(ele_n):
                pair_name = (ele_c, ele_n)
                pair_OS = (v_c, v_n)
                pair_CN = (CN_c, CN_n)
            if res.periodic_table.elements_list.index(ele_c) > res.periodic_table.elements_list.index(ele_n):
                pair_name = (ele_n, ele_c)
                pair_OS = (v_n, v_c)
                pair_CN = (CN_n, CN_c)
            if res.periodic_table.elements_list.index(ele_c) == res.periodic_table.elements_list.index(ele_n):
                if v_c <= v_n:
                    pair_name = (ele_c, ele_n)
                    pair_OS = (v_c, v_n)
                    pair_CN = (CN_c, CN_n)
                else:
                    pair_name = (ele_n, ele_c)
                    pair_OS = (v_n, v_c)
                    pair_CN = (CN_n, CN_c)
            CN_name = pair_CN
            #CN_name = SCN
            OS_name = pair_OS
            label = (CN_name, OS_name)
    
            if pair_name not in temp_pair_info:
                temp_pair_info[pair_name] = {}
                if label not in temp_pair_info[pair_name]:
                    temp_pair_info[pair_name][label] = [length_list[j]]
                else:
                    temp_pair_info[pair_name][label].append(length_list[j])
            else:
                if label not in temp_pair_info[pair_name]:
                    temp_pair_info[pair_name][label] = [length_list[j]]
                else:
                    temp_pair_info[pair_name][label].append(length_list[j])
    for k,v in temp_pair_info.items():
        for kk,vv in v.items():
            temp_pair_info[k][kk] = (sum(vv)/len(vv), len(vv))
    return temp_pair_info


def spider_pair_length_with_CN_unnorm(valence_list, res):
    temp_pair_info = {}
    for i in res.idx:
        length_list = res.matrix_of_length[i]
        for j in res.shell_idx_list[i]:
            ele_c = get_ele_from_sites(i,res)
            ele_n = get_ele_from_sites(j,res)
            v_c = str(valence_list[i])
            v_n = str(valence_list[j])
            CN_c = len(res.shell_ele_list[i])
            CN_n = len(res.shell_ele_list[j])
            SCN = CN_c + CN_n
            
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
            #CN_name = SCN
            OS_name = pair_OS
            label = (CN_name, OS_name)
    
            if pair_name not in temp_pair_info:
                temp_pair_info[pair_name] = {}
                if label not in temp_pair_info[pair_name]:
                    temp_pair_info[pair_name][label] = [length_list[j]]
                else:
                    temp_pair_info[pair_name][label].append(length_list[j])
            else:
                if label not in temp_pair_info[pair_name]:
                    temp_pair_info[pair_name][label] = [length_list[j]]
                else:
                    temp_pair_info[pair_name][label].append(length_list[j])
    return temp_pair_info


def normalization(single_result_dict):
    N = len([k for k,v in single_result_dict.items() if v != None])
    normalized_single_result_info = {}
    for i,info in single_result_dict.items():
        if info != None:
            for pair_name,label_info in info.items():
                if pair_name not in normalized_single_result_info:
                    normalized_single_result_info[pair_name] = {}
                    for label,length_info in label_info.items():
                        if label not in normalized_single_result_info[pair_name]:
                            normalized_single_result_info[pair_name][label] = []
                            normalized_single_result_info[pair_name][label].append(length_info)
                        else:
                            normalized_single_result_info[pair_name][label].append(length_info)
                else:
                    for label,length_info in label_info.items():
                        if label not in normalized_single_result_info[pair_name]:
                            normalized_single_result_info[pair_name][label] = []
                            normalized_single_result_info[pair_name][label].append(length_info)
                        else:
                            normalized_single_result_info[pair_name][label].append(length_info)
                
    for pair_name, info in normalized_single_result_info.items():
        for label, length_info in info.items():
            average_length = sum([(i[0]*i[1]) for i in length_info])/sum(i[1] for i in length_info)
            nomalized_num = sum(i[1] for i in length_info)/N
            normalized_single_result_info[pair_name][label] = (average_length, nomalized_num)
    return normalized_single_result_info


def global_normalization(pairs_info_dict):
    global_normalized_dict = {}
    for mid,info in pairs_info_dict.items():
        for pair_name,label_info in info.items():
            if pair_name not in global_normalized_dict:
                global_normalized_dict[pair_name] = {}
                for label,length_info in label_info.items():
                    if label not in global_normalized_dict[pair_name]:
                        global_normalized_dict[pair_name][label] = []
                        global_normalized_dict[pair_name][label].append(length_info)
                    else:
                        global_normalized_dict[pair_name][label].append(length_info)
            else:
                for label,length_info in label_info.items():
                    if label not in global_normalized_dict[pair_name]:
                        global_normalized_dict[pair_name][label] = []
                        global_normalized_dict[pair_name][label].append(length_info)
                    else:
                        global_normalized_dict[pair_name][label].append(length_info)
                        
    for pair_name, info in global_normalized_dict.items():
        for label, length_info in info.items():
            average_length = sum([(i[0]*i[1]) for i in length_info])/sum(i[1] for i in length_info)
            nomalized_num = sum(i[1] for i in length_info)
            global_normalized_dict[pair_name][label] = (average_length, nomalized_num)    
                        
    return global_normalized_dict


def classify(atom_idx_list,res):
    classes = []
    for i in atom_idx_list:
        group = "super" if i in res.super_atom_idx_list else "link"
        classes.append((get_ele_from_sites(i,res), tuple(sorted(res.shell_ele_list[i])), group))
    classes_dict = {}
    for i,c in zip(atom_idx_list, classes):
        if c in classes_dict:
            classes_dict[c].append(i)
        else:
            classes_dict[c] = [i]
    return classes_dict





def cal_loss_func_by_OLS(temp_pair_info,pred_dict):
    loss = 0
    N = 0
    for pair_name,info in temp_pair_info.items():
        if pair_name in pred_dict:
            for label, length_list in info.items():
                if label in pred_dict[pair_name]:
                    pred_length = pred_dict[pair_name][label][0]
                    loss_temp = sum([((l-pred_length)/pred_length)**2 for l in length_list])
                    N_temp = len(length_list)
                    loss += loss_temp
                    N += N_temp  

                else:
                    loss += len(length_list)
                    N += len(length_list)

        else:
            for label, length_list in info.items():
                loss += len(length_list)
                N += len(length_list)

    LOSS = loss/N * 10000
    return LOSS


def cut_the_work_list(meet, piece_length):
    Npiece = len(meet)//piece_length + 1
    work_list = []
    lp = 0
    rp = piece_length
    while Npiece != 1:
        work_list.append(meet[lp:rp])
        lp += piece_length
        rp += piece_length
        Npiece -= 1
    else:
        work_list.append(meet[lp:len(meet)])
    return work_list



def enu_permu_combi(limit):
    #make sure all the elements in limit greater than 0.
    for l in limit:
        if l < 0:
            raise ValueError
            break
    v = 1
    in_v = 1
    for i in limit:
        v = v * (i+1)
        in_v = in_v * (i+1)
    solution = []
    for i in range(in_v):
        line = [0 for j in range(len(limit))]
        solution.append(line)
    i = 0
    k = 0
    while i < len(limit):
        v = v/(limit[i] + 1)
        l = 0
        while l < in_v:
            j = 0
            while j < v:
                solution[l][i] = k
                j = j + 1
                l = l + 1
            k = k + 1
            if k > limit[i]:
                k = 0
        i = i + 1
    return solution


def spider_bond_length(res):
    temp_pair_info = {}
    for i in res.idx:
        length_list = res.matrix_of_length[i]
        for j in res.shell_idx_list[i]:
            ele_c = get_ele_from_sites(i,res)
            ele_n = get_ele_from_sites(j,res)
            v_c = str(res.sum_of_valence[i])
            v_n = str(res.sum_of_valence[j])
            CN_c = len(res.shell_ele_list[i])
            CN_n = len(res.shell_ele_list[j])
            SCN = CN_c + CN_n

            if res.periodic_table.elements_list.index(ele_c) < res.periodic_table.elements_list.index(ele_n):
                pair_name = (ele_c, ele_n)
                pair_OS = (v_c, v_n)
                pair_CN = (CN_c, CN_n)
            if res.periodic_table.elements_list.index(ele_c) > res.periodic_table.elements_list.index(ele_n):
                pair_name = (ele_n, ele_c)
                pair_OS = (v_n, v_c)
                pair_CN = (CN_n, CN_c)
            if res.periodic_table.elements_list.index(ele_c) == res.periodic_table.elements_list.index(ele_n):
                if v_c <= v_n:
                    pair_name = (ele_c, ele_n)
                    pair_OS = (v_c, v_n)
                    pair_CN = (CN_c, CN_n)
                else:
                    pair_name = (ele_n, ele_c)
                    pair_OS = (v_n, v_c)
                    pair_CN = (CN_n, CN_c)
            CN_name = pair_CN
            #CN_name = SCN
            OS_name = pair_OS
            label = (CN_name, OS_name)
    
            if pair_name not in temp_pair_info:
                temp_pair_info[pair_name] = {}
                if label not in temp_pair_info[pair_name]:
                    temp_pair_info[pair_name][label] = [length_list[j]]
                else:
                    temp_pair_info[pair_name][label].append(length_list[j])
            else:
                if label not in temp_pair_info[pair_name]:
                    temp_pair_info[pair_name][label] = [length_list[j]]
                else:
                    temp_pair_info[pair_name][label].append(length_list[j])
    return temp_pair_info


def cal_loss_func_by_MLE(temp_pair_info,pred_dict, global_sigma_dict, global_mean_dict):
    n = sum([len(vv) for v in temp_pair_info.values() for vv in v.values()])

    likelyhood = 0

    for pair_name,info in temp_pair_info.items():
        #NL = sum([i[0] for i in useful_pair.values()])
        if pair_name in pred_dict:
            useful_pair = pred_dict[pair_name]
            for label,length_list in info.items():

                key = (pair_name, label[0], label[1])
                try:
                    mean = round(global_mean_dict[key],3)
                    sigma = round(global_sigma_dict[key],3)
                    sigma = 0.01 if sigma == 0 else sigma
                except Exception as e:
                    #print(f"An error occurred: {e}")
                    possible_keys = [k for k in global_mean_dict.keys() if k[0] == pair_name]
                    mean = np.mean([global_mean_dict[key] for key in possible_keys])
                    sigma = np.mean([global_sigma_dict[key] for key in possible_keys])

                likelyhood_prime = 0
                for l in length_list:
                    gx = (1/(np.sqrt(2*np.pi)*sigma)) * np.exp(-(round(l,3)-mean)**2/(2*sigma**2))
                    gx_den = gx * 0.001
                    math_domin_limit = 10**(-323.60)
                    gx_den = gx_den if gx_den > math_domin_limit else math_domin_limit
                    likelyhood_prime += math.log(gx_den)
                likelyhood += likelyhood_prime
        else:
            raise ValueError("WRONG!")

    avg_likelyhood = -1*(sum_likelyhood/n)
    return avg_likelyhood


def cal_loss_func_by_OLS(temp_pair_info,pred_dict, global_sigma_dict, global_mean_dict):
    n = sum([len(vv) for v in temp_pair_info.values() for vv in v.values()])

    OLS = 0

    for pair_name,info in temp_pair_info.items():
        #NL = sum([i[0] for i in useful_pair.values()])
        if pair_name in pred_dict:
            useful_pair = pred_dict[pair_name]
            for label,length_list in info.items():
                key = (pair_name, label[0], label[1])
                try:
                    mean = round(global_mean_dict[key],3)
                except Exception as e:
                    #print(f"An error occurred: {e}")
                    possible_keys = [k for k in global_mean_dict.keys() if k[0] == pair_name]
                    mean = np.mean([global_mean_dict[key] for key in possible_keys])

                OLS_prime = 0
                for l in length_list:
                    OLS_prime += (round(l,3)-mean)**2
                OLS += OLS_prime
        else:
            raise ValueError("WRONG!")

    OLS = OLS/n
    return OLS


def cal_loss_func_by_MAP(temp_pair_info, pred_dict, global_sigma_dict, global_mean_dict):
    n = sum([len(vv) for v in temp_pair_info.values() for vv in v.values()])

    likelyhood = 0
    prior = 0

    for pair_name,info in temp_pair_info.items():
        #NL = sum([i[0] for i in useful_pair.values()])
        if pair_name in pred_dict:
            useful_pair = pred_dict[pair_name]
            for label,length_list in info.items():

                #it is the prior probability calculation:
                NL = sum([v[1] for k,v in useful_pair.items() if k[0] == label[0]])
                if NL == 0:
                    NL = sum([v[1] for k,v in useful_pair.items()])
                try:
                    nl = useful_pair[label][1]
                except Exception as e:
                    #print(f"MAP prior calculation error: {e}")
                    nl = 1
                prior_prime = len(length_list) * math.log(nl/NL)
                prior += prior_prime
                #print("likelyhood:%s"%likelyhood)

                #it is the likelyhood calculation:
                key = (pair_name, label[0], label[1])
                try:
                    mean = round(global_mean_dict[key],3)
                    sigma = round(global_sigma_dict[key],3)
                    sigma = 0.01 if sigma == 0 else sigma
                except Exception as e:
                    #print(f"MAP likelyhood calculation error: {e}")
                    possible_keys = [k for k in global_mean_dict.keys() if k[0] == pair_name]
                    mean = np.mean([global_mean_dict[key] for key in possible_keys])
                    sigma = np.mean([global_sigma_dict[key] for key in possible_keys])

                likelyhood_prime = 0
                for l in length_list:
                    gx = (1/(np.sqrt(2*np.pi)*sigma)) * np.exp(-(round(l,3)-mean)**2/(2*sigma**2))
                    gx_den = gx * 0.001
                    math_domin_limit = 10**(-323.60)
                    gx_den = gx_den if gx_den > math_domin_limit else math_domin_limit
                    likelyhood_prime += math.log(gx_den)
                likelyhood += likelyhood_prime

        else:
            for label,length_list in info.items():
                NL = 100000
                nl = 1
                prior_prime = len(length_list) * math.log(nl/NL)
                prior += prior_prime
                raise ValueError("WRONG!")

    avg_likelyhood = -1*(likelyhood/n)
    avg_prior = -1*(prior/n)
    MAP = avg_likelyhood + avg_prior
    #print("likelyhood: "+str(avg_likelyhood)+"   prior: "+str(avg_prior)+"   sum: "+str(avg_likelyhood+avg_prior))
    return MAP, avg_likelyhood, avg_prior


def global_normalization_sigma_mean(pairs_info_dict):
    global_normalized_dict = {}
    global_coeficient_dict = {}
    for k,v in pairs_info_dict.items():
        Nt = len(v)  
        for t, info in v.items():
            for pair, length in info.items():
                for label, l in length.items():
                    key = (pair, label[0], label[1])
                    coef = [(1/Nt) for i in range(len(l))]
                    if key not in global_normalized_dict:
                        global_normalized_dict[key] = l
                        global_coeficient_dict[key] = coef
                    else:
                        global_normalized_dict[key] += l 
                        global_coeficient_dict[key] += coef

    global_sigma_dict = {}
    global_mean_dict = {}
    for k,v in global_normalized_dict.items():
        coef = global_coeficient_dict[k]
        mean = np.divide(np.sum(np.multiply(np.array(v), coef)),np.sum(np.multiply(np.ones(len(v)), coef)))
        d = np.divide(np.sum(np.multiply(np.square(np.array(v)-mean), coef)),np.sum(np.multiply(np.ones(len(v)),coef)))
        std = np.sqrt(d)
        global_sigma_dict.update({k:round(std,3)})
        global_mean_dict.update({k:round(mean,3)})
    return global_normalized_dict, global_sigma_dict,global_mean_dict


def global_normalization_super_point(super_point_dict):
    global_point_normed_dict = {}
    for mid, info in super_point_dict.items():
        if info != None:
            Nt = len(info)
            for t, point_list in info.items():
                for point in point_list:
                    point = str(point)
                    if point in global_point_normed_dict:
                        global_point_normed_dict[point] += 1/Nt
                    else:
                        global_point_normed_dict[point] = 1/Nt
    return global_point_normed_dict