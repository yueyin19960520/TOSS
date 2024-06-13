import time
import pickle
import os
import random
import openpyxl
import argparse
import copy
import pandas as pd

from get_fos import GET_FOS
from result import RESULT
from pre_set import PRE_SET
from digest import DIGEST
from get_structure import GET_STRUCTURE
from post_process import *
from auxilary import *

from multiprocessing.dummy import Pool as ThreadPool
from functools import partial 
import multiprocessing


############## Functions for Get valid Tolerance ##############
def get_the_valid_t(m_id,i, server=False, filepath="/"):
    res = RESULT()
    PS = PRE_SET(spider = False)
    res.dict_ele, res.matrix_of_threshold = PS.dict_ele, PS.matrix_of_threshold
    res.Forced_transfer_group,res.inorganic_group = PS.Forced_transfer_group,PS.inorganic_group

    if not server:
        GS = GET_STRUCTURE(m_id)
    else:
        filepath, m_id = os.path.split(filepath)

        GS = GET_STRUCTURE(m_id, specific_path="../toss_server/%s/"%filepath)
    res.sites,res.idx, res.struct = GS.sites, GS.idx, GS.struct
    res.matrix_of_length, res.valence_list, res.elements_list = GS.matrix_of_length, GS.valence_list, GS.elements_list

    valid_t = []
    check_result = []

    for t in [round(1.1 + 0.01 * i,2) for i in range(16)]:
        try:
            DG = DIGEST(valid_t, t, m_id, res)
            res.max_oxi_list, res.min_oxi_list = DG.max_oxi_list, DG.min_oxi_list
            res.SHELL_idx_list, res.threshold_list = DG.SHELL_idx_list, DG.threshold_list
            res.organic_patch, res.alloy_flag = DG.organic_patch, DG.alloy_flag
            DG.digest_structure_with_image(res)
            res.shell_ele_list, res.shell_env_list, res.shell_idx_list, res.shell_CN_list, res.shell_X_list = DG.shell_ele_list, DG.shell_env_list, DG.shell_idx_list, DG.shell_CN_list, DG.shell_X_list

            if res.shell_CN_list not in check_result:
                check_result.append(res.shell_CN_list)
                valid_t.append(t)
            else:
                None
        except Exception as e:  
            print(f"An error occurred: {e}")
            None

    if not server:
        print('This is the %sth structure with mid %s and we got %s different valid tolerance(s).'%(i, m_id, len(valid_t)))
        return valid_t
    else:
        print_info = 'The structure with mid %s has %s tolerance(s), which are %s.'%(m_id, len(valid_t), valid_t)
        return print_info, valid_t


def tolerance_corr(func, m_id, delta_X, tolerance_list):
    tolerance_trial = tolerance_list
    single_result_dict_normed = {}
    single_result_dict = {}
    single_super_point_dict = {}
    for t in tolerance_trial:
        try:
            res = RESULT()
            res.mid = m_id
            func(m_id, delta_X, t, tolerance_list, res)
            temp_pair_info_normed = spider_pair_length_with_CN_normed(res)
            temp_pair_info = spider_bond_length(res)
            single_result_dict[t] = temp_pair_info
            single_result_dict_normed[t] = temp_pair_info_normed
            super_point_list = [[get_ele_from_sites(i,res), sorted(res.shell_ele_list[i])] for i in res.idx]
            single_super_point_dict[t] = super_point_list
        except Exception as e:  
            print(f"An error occurred: {e}")
            LOSS = None
            temp_pair_info = None
            temp_pair_info_normed = None
            super_point_list = None
    return single_result_dict_normed, single_result_dict, single_super_point_dict #They could be empty.


def get_Initial_Guess(m_id, i):
    GFOS = GET_FOS()
    delta_X = 0.1

    #tolerance_list = get_the_valid_t(m_id, i)      
    tolerance_list = valid_t_dict[m_id]

    ls = time.time()
    try:
        single_result_dict_normed, single_result_dict, single_super_point_dict = tolerance_corr(GFOS.initial_guess, 
                                                                                                m_id, 
                                                                                                delta_X, 
                                                                                                tolerance_list)

        if single_result_dict_normed != {}:
            normalized_single_result_info = normalization(single_result_dict_normed)
        else:
            raise ValueError
        parameters = [m_id, tolerance_list, normalized_single_result_info, single_result_dict]
        tc = time.time() - ls
        print("Got the Formal Oxidation State of the %sth structure %s in %s seconds."%(i,m_id,tc))
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        return parameters
    except Exception as e:   
        print(f"An error occurred: {e}")
        parameters = [m_id, tolerance_list, None, None]
        tc = time.time() - ls
        print("Failed to analyze the %sth structure %s in %s seconds."%(i,m_id,tc))
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        return parameters


def assemble(parameters):

    if parameters[2] == None:
        failed_structure.append(parameters[0])
    elif parameters[2] == "RERUN":
        stuked_structure.append(parameters[0])
    else:
        succed_structure.append(parameters[0])
        pairs_info_normed_dict[parameters[0]] = parameters[2]
        pairs_info_dict[parameters[0]] = parameters[3]

    valid_t_dict[parameters[0]] = parameters[1]


    finished_len = len(succed_structure) + len(failed_structure) + len(stuked_structure)
    print("Successed_Structure_Number:%s, Failed_Structure_Number:%s, Stucked_Structure_Number:%s"%(len(succed_structure),len(failed_structure),len(stuked_structure)))
    print("The process has finished %s/%s."%(finished_len, len(target_group)))


def abortable_worker(func, *args, **kwargs):
    timeout = kwargs.get('timeout', None)
    p = ThreadPool(1)
    res = p.apply_async(func, args = args)
    try:
        out = res.get(timeout)
    except multiprocessing.TimeoutError:
        p.terminate()
        print("Aborting due to timeout.")
        print("Failed to analyze the structure %s within limit time."%args[0])
        out = [args[0], [], "RERUN", []]
    finally:
        p.close()
        p.join()
    return out

#############################################################################################
######################################GLOBAL VARIABLES#######################################
#############################################################################################

path = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]

succed_structure = []
failed_structure = []
stuked_structure = []

#valid_t_dict = {}
file_get = open("../valid_t_dict.pkl",'rb') 
valid_t_dict = pickle.load(file_get) 
file_get.close()

pairs_info_dict = {}
pairs_info_normed_dict = {}


target_group = os.listdir("D:/share/TOSS_2024/structures/")#os.listdir(os.path.join(path,"structures"))
random.shuffle(target_group)
NP = 600000

##########################################################################################################################
##########################################################################################################################
##########################################################################################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--ncore', type=int, required=True, help='Number of Parallel')
    parser.add_argument('-t1', '--timeout1', type=int, default=3000, help="The primary timeout seconds for each subprocess.")
    parser.add_argument('-t2', '--timeout2', type=int, default=180, help="The secondary timeout seconds for each subprocess.")
    args = parser.parse_args()

    start_time = time.time()
    spider_valid = True

    work_list = cut_the_work_list(target_group, NP)

    n = args.ncore
    primary_limit_time = args.timeout1
    secondary_limit_time = args.timeout2

    ######################################################################################################################
    ################################ Do the Initial Guess of the Oxidation States Result #################################
    ######################################################################################################################

    succed_structure, failed_structure, stuked_structure = [], [], []

    print("This run contains %d structures."%len(target_group))
    print("Main is processing......")

    for idx_of_list, sub_work_list in enumerate(work_list):
        pool = multiprocessing.Pool(n)
        for i, mid in enumerate(sub_work_list):
            abortable_func = partial(abortable_worker, get_Initial_Guess, timeout=primary_limit_time)
            pool.apply_async(abortable_func, args = (mid,i,), callback = assemble)
        pool.close()
        pool.join()

    print("Let's run it again!")

    rework_list = copy.deepcopy(stuked_structure)
    stuked_structure = []

    pool = multiprocessing.Pool(n)
    for i, mid in enumerate(rework_list):
        abortable_func = partial(abortable_worker, get_Initial_Guess, timeout=secondary_limit_time)
        pool.apply_async(abortable_func, args = (mid,i,), callback = assemble)
    pool.close()
    pool.join()

    print("Main is done!")
    end_time = time.time()
    cost = end_time - start_time
    print("The main processed %d samples by cost of %d seconds."%(len(target_group), cost))

    #file_save = open(path + "/valid_t_dict.pkl",'wb') 
    #pickle.dump(valid_t_dict, file_save) 
    #file_save.close()

    file_save = open(path + "/pairs_info_normed_dict.pkl",'wb') 
    pickle.dump(pairs_info_normed_dict, file_save) 
    file_save.close()

    file_save = open(path + "/pairs_info_dict.pkl",'wb') 
    pickle.dump(pairs_info_dict, file_save) 
    file_save.close()

    global_normalized_normed_dict = global_normalization(pairs_info_normed_dict)

    file_save = open(path + "/global_normalized_normed_dict_loop_0.pkl",'wb') 
    pickle.dump(global_normalized_normed_dict, file_save) 
    file_save.close()

    global_normalized_dict, global_sigma_dict, global_mean_dict = global_normalization_sigma_mean(pairs_info_dict)

    file_save = open(path + "/global_normalized_dict_loop_0.pkl",'wb') 
    pickle.dump(global_normalized_dict, file_save) 
    file_save.close()

    file_save = open(path + "/global_sigma_dict_loop_0.pkl",'wb') 
    pickle.dump(global_sigma_dict, file_save) 
    file_save.close()

    file_save = open(path + "/global_mean_dict_loop_0.pkl",'wb') 
    pickle.dump(global_mean_dict, file_save) 
    file_save.close()

    try:
        sent_message()
    except:
        None