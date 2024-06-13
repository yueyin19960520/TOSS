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
from initialization import INITIAL
from first_algo import FIRST_ALGO
from second_algo import SECOND_ALGO
from resonance import RESONANCE
from tune import TUNE
from post_process import *
from auxilary import *

from multiprocessing.dummy import Pool as ThreadPool
from functools import partial 
import multiprocessing


##############################################################################################################################################################
########################################################## Do the Initial Guess For TOSS #####################################################################
##############################################################################################################################################################


def get_Oxidation_States(m_id,i, atom_pool, server=False, filepath="/", input_tolerance_list=[]):
    if not server:
        tolerance_list = valid_t_dict[m_id]
    else:
        tolerance_list = input_tolerance_list

    GFOS = GET_FOS()
    delta_X = 0.1
    corr_t = []
    ls = time.time()
        
    for t in tolerance_list:
        res = RESULT()
        res.mid = m_id
        TN = TUNE()
        #try:
        if True:
            if not server:
                GFOS.loss_loop(m_id, delta_X, t, tolerance_list, res)
            else:
                res = GFOS.loss_loop(m_id, delta_X, t, tolerance_list, res=None, server=server, filepath=filepath)
            temp_pair_info = spider_pair_length_with_CN_unnorm(res.sum_of_valence, res)

            #now, the matched dict is the global normalization normed dict. 
            loss, likelyhood, prior = cal_loss_func_by_MAP(temp_pair_info, 
                                                           global_normalized_normed_dict, 
                                                           global_sigma_dict, 
                                                           global_mean_dict)
            N_spec = len(res.species_uni_list)
            res.initial_vl = res.sum_of_valence
            
            if len(res.super_atom_idx_list) > 0:
                if res.resonance_flag:
                    avg_LOSS, the_resonance_result = TN.tune_by_resonance(loss,
                                                                          likelyhood,
                                                                          prior,
                                                                          res, 
                                                                          global_normalized_normed_dict,
                                                                          global_sigma_dict, 
                                                                          global_mean_dict)
                    res.final_vl = the_resonance_result[0][0]
                    same_after_resonance = True if res.final_vl == res.initial_vl else False
                    res.sum_of_valence = res.final_vl

                if atom_pool == "super":
                    process_atom_idx_list = res.super_atom_idx_list
                if atom_pool == "link":
                    process_atom_idx_list = res.link_atom_idx_list
                if atom_pool == "all":
                    process_atom_idx_list = res.idx 

                LOSS, LIKELYHOOD, PRIOR, res.final_vl = TN.tune_by_redox_in_certain_range_by_MAP(process_atom_idx_list, 
                                                                              loss, 
                                                                              likelyhood,
                                                                              prior,
                                                                              res.sum_of_valence,
                                                                              0,
                                                                              res,
                                                                              global_normalized_normed_dict,
                                                                              global_sigma_dict, 
                                                                              global_mean_dict)

                res.sum_of_valence = res.final_vl
                same_after_tunation = True if res.final_vl == res.initial_vl else False
                same_after_resonance = True   
            else:
                res.final_vl = res.initial_vl
                same_after_tunation = True
                same_after_resonance = True
                LOSS = loss
                
            parameters = {m_id: [res.resonance_flag, same_after_tunation, same_after_resonance]}
            
            coef = 1.2
            loss_value = coef**N_spec * LOSS
            corr_t.append((t,loss_value,res))
        #except:
        else:
            None
    print(corr_t)
    #try:
    if True:
        chosen_one = sorted(corr_t, key = lambda x:x[1])[0]
        res = chosen_one[2]
        t = chosen_one[0]
        #parameters = [m_id, LOSS, res.final_vl]
        #for loop
        temp_pair_info_normed = spider_pair_length_with_CN_normed(res)
        temp_pair_info = spider_bond_length(res)

        single_result_dict_normed = {t:temp_pair_info_normed}
        single_result_dict = {t:temp_pair_info}

        OS_result_with_ele = sorted([(i,j) for i,j in zip(res.final_vl, res.elements_list)], key = lambda x :x[1])
        OS_result = [ij[0] for ij in OS_result_with_ele]

        normalized_single_result_info = normalization(single_result_dict_normed)
        parameters = [m_id, normalized_single_result_info, single_result_dict, OS_result]
        tc = time.time() - ls
        print("Got the Formal Oxidation State of the %sth structure %s in %s seconds."%(i,m_id,tc))
        #print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        #return parameters
    #except:
    else:
        parameters = [m_id, None, None, None]
        tc = time.time() - ls
        print("Failed to analyze the %sth structure %s in %s seconds."%(i,m_id,tc))

    if not server:
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        return parameters
    else:
        result = pd.DataFrame(np.vstack([np.array(res.elements_list),np.array(res.sum_of_valence),np.array(res.shell_CN_list)]))
        result.index = ["Elements", "Valence","Coordination Number"]
        return result
        


def assemble_OS(parameters):

    if parameters[1] != "RERUN":
        if parameters[1] != None:
            pairs_info_dict[parameters[0]] = parameters[2]
            pairs_info_normed_dict[parameters[0]] = parameters[1]
            OS_result_dict[parameters[0]] = parameters[3]
            succed_structure.append(parameters[0])
        else:
            failed_structure.append(parameters[0])
    else:
        stuked_structure.append(parameters[0])

    finished_len = len(succed_structure) + len(failed_structure) + len(stuked_structure)
    print("Successed_Structure_Number:%s, Failed_Structure_Number:%s, Stucked_Structure_Number:%s"%(len(succed_structure),len(failed_structure),len(stuked_structure)))
    print("The process has finished %s/%s."%(finished_len, len(target_group)))


def abortable_worker_OS(func, *args, **kwargs):
    timeout = kwargs.get('timeout', None)
    p = ThreadPool(1)
    res = p.apply_async(func, args = args)
    try:
        out = res.get(timeout)
    except multiprocessing.TimeoutError:
        p.terminate()
        print("Aborting due to timeout.")
        print("Failed to analyze the structure %s within limit time."%args[0])
        out = [args[0], "RERUN", "RERUN", "RERUN"]
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
pairs_info_dict = {}
pairs_info_normed_dict = {}
OS_result_dict = {}

file_get= open(path + "/global_normalized_normed_dict.pkl","rb")
global_normalized_normed_dict = pickle.load(file_get)
file_get.close()

file_get= open(path + "/global_mean_dict.pkl","rb")
global_mean_dict = pickle.load(file_get)
file_get.close()

file_get= open(path + "/global_sigma_dict.pkl","rb")
global_sigma_dict = pickle.load(file_get)
file_get.close()

file_get = open(path + "/valid_t_dict.pkl",'rb') 
valid_t_dict = pickle.load(file_get) 
file_get.close()

target_group = os.listdir(os.path.join(path,"structures"))
random.shuffle(target_group)
NP = 200000

##########################################################################################################################
##########################################################################################################################
##########################################################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--ncore', type=int, required=True, help='Number of Parallel')
    parser.add_argument('-t1', '--timeout1', type=int, default=1800, help="The primary timeout seconds for each subprocess.")
    parser.add_argument('-t2', '--timeout2', type=int, default=1800, help="The secondary timeout seconds for each subprocess.")
    args = parser.parse_args()

    start_time = time.time()
    spider_valid = True

    work_list = cut_the_work_list(target_group, NP)

    n = args.ncore
    primary_limit_time = args.timeout1
    secondary_limit_time = args.timeout2

    ######################################################################################################################
    ###################################### Start the Loop of the Oxidation States Tuning #################################
    ######################################################################################################################

    start_time = time.time()
    loop = "loop_1"
    
    print("This run contains %d structures."%len(target_group))
    print("Main is processing......")
    
    atom_pool = "all"
    rate_list = [0.5]

    while True:
        for idx_of_list, sub_work_list in enumerate(work_list):
            pool = multiprocessing.Pool(n)
            for i, m_id in enumerate(sub_work_list):
                #parameters = get_Oxidation_States(m_id,i,atom_pool)
                #assemble_OS(parameters)
                abortable_func = partial(abortable_worker_OS, get_Oxidation_States, timeout=primary_limit_time)
                pool.apply_async(abortable_func, args = (m_id,i,atom_pool), callback = assemble_OS)
            pool.close()
            pool.join()

        print("Let's run it again!")

        rework_list = copy.deepcopy(stuked_structure)
        stuked_structure = []
        
        pool = multiprocessing.Pool(n)
        for i, m_id in enumerate(rework_list):
            abortable_func = partial(abortable_worker_OS, get_Oxidation_States, timeout=secondary_limit_time)
            pool.apply_async(abortable_func, args = (m_id,i,atom_pool), callback = assemble_OS)
        pool.close()
        pool.join()

        print("Main is done!")
        end_time = time.time()
        cost = end_time - start_time
        print("The main processed %d samples by cost of %d seconds."%(len(target_group), cost))

        target_group = succed_structure + failed_structure
        succed_structure, failed_structure, stuked_structure = [],[],[] 

        global_normalized_normed_dict = global_normalization(pairs_info_normed_dict)
        file_save = open("../global_normalized_normed_dict_%s.pkl"%loop,'wb') 
        pickle.dump(global_normalized_normed_dict, file_save) 
        file_save.close()

        global_normalized_dict, global_sigma_dict, global_mean_dict = global_normalization_sigma_mean(pairs_info_dict)

        file_save = open(path + "/global_sigma_dict_%s.pkl"%loop,'wb') 
        pickle.dump(global_sigma_dict, file_save) 
        file_save.close()

        file_save = open(path + "/global_mean_dict_%s.pkl"%loop,'wb') 
        pickle.dump(global_mean_dict, file_save) 
        file_save.close()

        file_save = open(path + "/OS_result_dict_%s.pkl"%loop,'wb') 
        pickle.dump(OS_result_dict, file_save) 
        file_save.close()

        if loop != "loop_1":
            f_loop = "loop_" + str(int(loop[-1])-1)

            file_get = open(path + "/OS_result_dict_%s.pkl"%f_loop,'rb') 
            former_OS_result_dict = pickle.load(file_get) 
            file_get.close()

            S,s = 0,0
            for k,v in former_OS_result_dict.items():
                if k in OS_result_dict:
                    S += 1
                    if OS_result_dict[k] == v:
                        s += 1
            rate = round(s/S,4)

            try:
                sent_message(value3 = "atom_pool:%s; %s/%s=%s"%(atom_pool, s,S,rate))
            except:
                None
                
        else:
            rate = 0

        if atom_pool == "all" and rate >= 0.99:
            break
        elif abs(rate_list[-1] - rate) <= 0.005:
            break
        else:
            loop = "loop_" + str(int(loop[-1])+1)
            rate_list.append(rate)
            continue