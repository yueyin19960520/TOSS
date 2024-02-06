from result import RESULT
from pre_set import PRE_SET
from digest import DIGEST
from get_structure import GET_STRUCTURE
import time
import pickle
import os
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial 
import multiprocessing
from digest import get_ele_from_sites
from post_process import *
import sys


def classify_length(res):
    pair_dict = {}
    for i in res.idx:
        for j in res.shell_idx_list[i]:
            ele_i = get_ele_from_sites(i,res)
            ele_j = get_ele_from_sites(j,res)

            if res.periodic_table.elements_list.index(ele_i) < res.periodic_table.elements_list.index(ele_j):
                pair_name = (ele_i, ele_j)
            else:
                pair_name = (ele_j, ele_i)

            if pair_name not in pair_dict:
                pair_dict[pair_name] = [res.matrix_of_length[i][j]]
            else:
                pair_dict[pair_name].append(res.matrix_of_length[i][j])
    return pair_dict


def save_bond_length(pair_name, length_list, save_path):
    with open(save_path + "bond_of_%s_%s.csv"%(pair_name[0],pair_name[1]), "a+") as f:
        for l in length_list:
            f.write(str(l)+"\n")
        f.close()


def global_classify(pair_info_dict):
    global_pairs_info = {}
    for k, pair_info in pair_info_dict.items():
        for pair_name, length_list in pair_info.items():
            if pair_name not in global_pairs_info:
                global_pairs_info[pair_name] = length_list
            else:
                global_pairs_info[pair_name] += length_list
    return global_pairs_info


def Spider(m_id,i,t):
    res = RESULT()
    PS = PRE_SET(spider = True)
    res.dict_ele, res.matrix_of_threshold = PS.dict_ele, PS.matrix_of_threshold
    res.Forced_transfer_group,res.inorganic_group = PS.Forced_transfer_group,PS.inorganic_group
    #res.transit_metals, res.metals = PS.transit_matals, PS.metals

    GS = GET_STRUCTURE(m_id)
    res.sites,res.idx, res.struct = GS.sites, GS.idx, GS.struct
    res.matrix_of_length, res.valence_list, res.elements_list = GS.matrix_of_length, GS.valence_list, GS.elements_list

    valid_t = [1.1] if t == 1.1 else [1.1, t]
    DG = DIGEST(valid_t, t, m_id, res)
    res.max_oxi_list, res.min_oxi_list = DG.max_oxi_list, DG.min_oxi_list
    res.SHELL_idx_list, res.threshold_list = DG.SHELL_idx_list, DG.threshold_list
    res.organic_patch, res.alloy_flag = DG.organic_patch, DG.alloy_flag
    DG.digest_structure_with_image(res)
    res.shell_ele_list, res.shell_env_list, res.shell_idx_list, res.shell_CN_list, res.shell_X_list = DG.shell_ele_list, DG.shell_env_list, DG.shell_idx_list, DG.shell_CN_list, DG.shell_X_list

    pair_dict = classify_length(res)
    parameter = [m_id, pair_dict]
    print('This is the %sth structure with mid %s and we got %s different pairs.'%(i, m_id, len(pair_dict)))
    return parameter


def Global_Spider(m_id,i,t):
    res = RESULT()
    PS = PRE_SET(spider = True, work_type = "global")
    res.dict_ele, res.matrix_of_threshold = PS.dict_ele, PS.matrix_of_threshold
    res.Forced_transfer_group,res.inorganic_group = PS.Forced_transfer_group,PS.inorganic_group
    #res.transit_metals, res.metals = PS.transit_matals, PS.metals

    GS = GET_STRUCTURE(m_id)
    res.sites,res.idx, res.struct = GS.sites, GS.idx, GS.struct
    res.matrix_of_length, res.valence_list, res.elements_list = GS.matrix_of_length, GS.valence_list, GS.elements_list
    
    valid_t = [1.1] if t == 1.1 else [1.1, t]
    DG = DIGEST(valid_t, t, m_id, res)
    res.max_oxi_list, res.min_oxi_list = DG.max_oxi_list, DG.min_oxi_list
    res.SHELL_idx_list, res.threshold_list = DG.SHELL_idx_list, DG.threshold_list
    res.organic_patch, res.alloy_flag = DG.organic_patch, DG.alloy_flag
    DG.digest_structure_with_image(res)
    res.shell_ele_list, res.shell_env_list, res.shell_idx_list, res.shell_CN_list, res.shell_X_list = DG.shell_ele_list, DG.shell_env_list, DG.shell_idx_list, DG.shell_CN_list, DG.shell_X_list

    pair_dict = classify_length(res)
    parameter = [m_id, pair_dict]
    print('This is the %sth structure with mid %s and we got %s different pairs.'%(i, m_id, len(pair_dict)))
    return parameter
"""END HERE"""