import time
import pickle
import os
import sys
import copy
import numpy as np
import pandas as pd
import argparse
import random
import math
import re

from result import RESULT
from pre_set import PRE_SET
from digest import DIGEST
from get_structure import GET_STRUCTURE
from digest import get_ele_from_sites
from post_process import *
from spider_length import *
from fitting import *

from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
from functools import partial 
import multiprocessing

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec



##############################################################################################################################################################
######################################################### Calculate the Unique Tolerance #####################################################################
##############################################################################################################################################################
def get_unique_tolerance(m_id,i):
    res = RESULT()
    PS = PRE_SET(spider = True, work_type = "global")
    res.dict_ele, res.matrix_of_threshold = PS.dict_ele, PS.matrix_of_threshold
    res.Forced_transfer_group,res.inorganic_group = PS.Forced_transfer_group,PS.inorganic_group

    GS = GET_STRUCTURE(m_id)
    res.sites,res.idx, res.struct = GS.sites, GS.idx, GS.struct
    res.matrix_of_length, res.valence_list, res.elements_list = GS.matrix_of_length, GS.valence_list, GS.elements_list

    valid_t = []
    t_with_nos = []
    check_result = []

    for t in [round(1.1 + 0.01 * i,2) for i in range(16)]: #from 1.1 to 1.25
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
                species_list = [(i,j) for i,j in zip(res.elements_list, res.shell_env_list)]
                num_of_species = len(set(species_list))
                t_with_nos.append((t,num_of_species))
        except:
            None

    min_num_of_species = min([n[1] for n in t_with_nos])
    unique_t = max([n[0] for n in t_with_nos if n[1] == min_num_of_species])

    parameter = [m_id, unique_t]
    print('This is the %sth structure with mid %s and we got the unique tolerance is %s.'%(i, m_id, unique_t))
    return parameter


def abortable_worker_unique_t(func, *args, **kwargs):
    timeout = kwargs.get('timeout', None)
    p = ThreadPool(1)
    res = p.apply_async(func, args = args)
    try:
        out = res.get(timeout)
    except multiprocessing.TimeoutError:
        p.terminate()
        print("Aborting due to timeout.")
        print("Failed to analyze the structure %s within limit time."%args[0])
        out = [args[0], "RERUN"]
    finally:
        p.close()
        p.join()
    return out


def assemble_unique_t(parameter):
    global tolerance_dict
    global stuked_structures
    if parameter[1] == "RERUN":
        stuked_structures.append(parameter[0])
        tolerance_dict[parameter[0]] = []
    else:
        succed_structures.append(parameter[0])
        tolerance_dict[parameter[0]] = parameter[1]

    finished_len = len(succed_structures) +  len(stuked_structures)
    print("Successed_Structure_Number:%s, Stucked_Structure_Number:%s"%(len(succed_structures),len(stuked_structures)))
    print("The process has finished %s/%s."%(finished_len, len(target_group)))


##############################################################################################################################################################
######################################################### SAVE ALL THE BOND LENGTH MATRIX ####################################################################
##############################################################################################################################################################
def get_length_matrix(mid, i):
    try:
        res = RESULT()
        GS = GET_STRUCTURE(mid)
        res.sites,res.idx, res.struct = GS.sites, GS.idx, GS.struct
        res.matrix_of_length, res.valence_list, res.elements_list = GS.matrix_of_length, GS.valence_list, GS.elements_list
        np.fill_diagonal(res.matrix_of_length, 0)
        parameters = [mid, "GOOD", res.matrix_of_length]
    except:
        parameters = [mid, None, None]
    return parameters


def assemble_LM(parameters):
    global length_matrix_dict
    global failed_structure
    global stuked_structure

    if parameters[1] != "RERUN":
        if parameters[1] != None:
            length_matrix_dict.update({parameters[0]: parameters[2]})
        else:
            failed_structure.append(parameters[0])
    else:
        stuked_structure.append(parameters[0])

    finished_len = len(length_matrix_dict) + len(failed_structure) + len(stuked_structure)
    print("Successed_Structure_Number:%s, Failed_Structure_Number:%s, Stucked_Structure_Number:%s"%(len(length_matrix_dict),len(failed_structure),len(stuked_structure)))
    print("The process has finished %s/%s."%(finished_len, len(target_group)))


def abortable_worker_LM(func, *args, **kwargs):
    timeout = kwargs.get('timeout', None)
    p = ThreadPool(1)
    res = p.apply_async(func, args = args)
    try:
        out = res.get(timeout)
        return out
    except multiprocessing.TimeoutError:
        p.terminate()
        print("Aborting due to timeout.")
        print("Failed to analyze the structure %s within limit time."%args[0])
        out = [args[0],"RERUN", None]
    finally:
        p.close()
        p.join()
    return out



##############################################################################################################################################################
############################################################### Matrix of Threshold Loop  ####################################################################
##############################################################################################################################################################
def assemble_looping_data(parameters): #result_dict
    global result_dict
    if parameters[1] != "None":
        print(parameters[1])
        result_dict[parameters[0]] = parameters[1]
        print("The whole process finished %s calculations."%len(result_dict))


def assemble_length_data(parameter): #pair_info_dict
    global pair_info_dict
    if len(parameter) == 2:
        pair_info_dict[parameter[0]] = parameter[1]    
    print("The process has finished %s/%s."%(len(pair_info_dict), len(target_group)))


def assemble_final_data(parameters): #fitting_info_dict
    global fitting_info_dict
    if len(parameters) == 2:
        fitting_info_dict[parameters[0]] = parameters[1]
    print("The process has finished %s/%s."%(len(fitting_info_dict), len(target_group)))


def abortable_worker_MOT(func, *args, **kwargs):
    timeout = kwargs.get('timeout', None)
    p = ThreadPool(1)
    res = p.apply_async(func, args = args)
    try:
        out = res.get(timeout)
        return out
    except multiprocessing.TimeoutError:
        p.terminate()
        print("Aborting due to timeout.")
        print("Failed to analyze the structure %s within limit time."%args[0])
        out = [args[0]]
    finally:
        p.close()
        p.join()
    return out


def renew_threshold_matrix(result_dict, renew_path):
    threshold_data = []
    for i,ele1 in enumerate(element_list):
        temp_list = []
        for j,ele2 in enumerate(element_list):
            if (ele1, ele2) in result_dict:
                temp_list.append(result_dict[(ele1,ele2)]["threshold"])
            elif (ele2, ele1) in result_dict:
                temp_list.append(result_dict[(ele2,ele1)]["threshold"])
            else:
                temp_list.append(0)
        threshold_data.append(temp_list)

    threshold_df = pd.DataFrame(data=threshold_data,index=element_list,columns=element_list)
    threshold_df.to_csv(renew_path + "/threshold_matrix_looping.csv")
    threshold_df.to_csv(renew_path + "/threshold_matrix_%s.csv"%loop)
    return threshold_data


def save_final_threshold_matrix(fitting_info_dict, main_path):
    threshold_data = []
    for i,ele1 in enumerate(element_list):
        temp_list = []
        for j,ele2 in enumerate(element_list):
            if (ele1, ele2) in fitting_info_dict:
                temp_list.append(fitting_info_dict[(ele1,ele2)]["threshold"])
            elif (ele2, ele1) in fitting_info_dict:
                temp_list.append(fitting_info_dict[(ele2,ele1)]["threshold"])
            else:
                temp_list.append(0)
        threshold_data.append(temp_list)

    threshold_df = pd.DataFrame(data=threshold_data,index=element_list,columns=element_list)
    threshold_df.to_csv(main_path + "threshold_matrix_loopped.csv")
    return None


def check_convergence(all_result,threshold_data,list_of_PID):
    if len(all_result) == 0:
        for i in range(118):
            for j in range(118):
                if i>=j:
                    if threshold_data[i][j] != 0:
                        if (i,j) not in all_result:
                            all_result[(i,j)] = [round(threshold_data[i][j], 3)]

    else:
        for i in range(118):
            for j in range(118):
                if (i,j) in all_result:
                    all_result[(i,j)].append(round(threshold_data[i][j], 3))

    con = 0
    osc = 0
    if len(list_of_PID) > 10:
        for k,v in all_result.items():
            if v[-1] == v[-2]:
                con += 1
            last = v.pop(-1)
            if last in v:
                osc += 1
            v.append(last)
    return con, osc, all_result


def check_convergence_new(all_result, threshold_data, loop):
    if len(all_result) == 0:
        for i in range(118):
            for j in range(118):
                if i>=j:
                    if threshold_data[i][j] != 0:
                        if (i,j) not in all_result:
                            all_result[(i,j)] = [round(threshold_data[i][j], 3)]

    else:
        for i in range(118):
            for j in range(118):
                if (i,j) in all_result:
                    all_result[(i,j)].append(round(threshold_data[i][j], 3))

    
    if loop > 3:
        plotting_info = get_plotting_info(all_result)
        if len(plotting_info) > len(all_result) * 0.999:
            converged = True
        else:
            converged = False
    else:
        converged = False
        plotting_info = {}

    return converged, plotting_info, all_result


def find_length(l, lp):
    group = []
    for i in range(0,len(l)-lp+1):
        temp_l = []
        for j in range(i,i+lp):
            temp_l.append(l[j])
        temp_l = tuple(temp_l)
        group.append(temp_l)
    flag = True
    if len(set(group)) != len(group):
        for i,p1 in enumerate(group):
            for j,p2 in enumerate(group):
                if flag:
                    if j > i: 
                        if p1 == p2:
                            s2e = (i,j)
                            flag = False
    return s2e


def get_plotting_info(all_result):
    plotting_info = {}
    for k,v in all_result.items():
        try:
            s2e = find_length(v, 2)
            candi = [(v[i], i) for i in range(s2e[0], s2e[1])]
            scandi = sorted(candi, key = lambda x:x[0])
            slt = scandi[-1][1]
            plotting_info[(element_list[k[0]],element_list[k[1]])] = slt
        except:
            s2e = None
    return plotting_info



####################################################################################################
#########################################GLOBAL VARIABLES###########################################
####################################################################################################
path = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
target_group = os.listdir(path + "/structures/")

NP = 200000
save_intermediate_variable = True

tolerance_dict = {}
succed_structures = []
stuked_structures = []

length_matrix_dict = {}
failed_structure = []
stuked_structure = []
####################################################################################################
####################################################################################################
####################################################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--ncore', type=int, required=True, help='Number of Parallel')
    parser.add_argument('-l', '--loop', type=int, default=20, help='max loop for Matrix of threshold.')
    parser.add_argument('-t1', '--timeout1', type=int, default=300, help="The primary timeout seconds for each subprocess.")
    parser.add_argument('-t2', '--timeout2', type=int, default=1800, help="The secondary timeout seconds for each subprocess.")
    args = parser.parse_args()

    n = args.ncore
    timeout1 = args.timeout1
    timeout2 = args.timeout2
    max_loop = args.loop


    element_list = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 
                'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga','Ge', 'As', 
                'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 
                'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 
                'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 
                'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 
                'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh',
                'Fl', 'Mc', 'Lv', 'Ts', 'Og']

    work_list = cut_the_work_list(target_group, NP)

    print("This run contains %d structures."%len(target_group))
    print("Main is processing......")

    for idx_of_list, sub_work_list in enumerate(work_list):
        pool = multiprocessing.Pool(n)
        for i, mid in enumerate(sub_work_list):
            abortable_func = partial(abortable_worker_unique_t, get_unique_tolerance, timeout=timeout1)
            pool.apply_async(abortable_func, args = (mid,i,), callback = assemble_unique_t)
        pool.close()
        pool.join()

    print("Main is done! But there still are some structures failed, let's do it again!")

    rerun_structures = copy.deepcopy(stuked_structures)
    stuked_structures = []

    pool = multiprocessing.Pool(n)
    for i, mid in enumerate(rerun_structures):
        abortable_func = partial(abortable_worker_unique_t, get_unique_tolerance, timeout=timeout2)
        pool.apply_async(abortable_func, args = (mid,i,), callback = assemble_unique_t)
    pool.close()
    pool.join()

    print("No matter how many failed, let us stop here and check the result!")
    file_save = open(path + "/global_tolerance_dict.pkl",'wb') 
    pickle.dump(tolerance_dict, file_save) 
    file_save.close()

    print("Secondary Main is processing......")
    
    for idx_of_list, sub_work_list in enumerate(work_list):
        pool = multiprocessing.Pool(n)
        for i, mid in enumerate(sub_work_list):
            abortable_func = partial(abortable_worker_LM, get_length_matrix, timeout=timeout1)
            pool.apply_async(abortable_func, args = (mid,i,), callback = assemble_LM)
        pool.close()
        pool.join()
    

    print("ALL DONE!")
    for k,v in length_matrix_dict.items():
        np.fill_diagonal(v, 0)
    file_save = open(path + "/length_matrix_dict.pkl",'wb') 
    pickle.dump(length_matrix_dict, file_save) 
    file_save.close()


    ######################################################################################################################
    ########################### INITIAL GUESS THE THRESHOLD AND SAVED THE INITIAL LOOPING ################################
    ######################################################################################################################

    openexcel = pd.read_excel(path + '/pre_set.xlsx', sheet_name = "Radii_X")
    radius = openexcel["single"].tolist()
    temp_matrix = []
    for i in range(len(radius)):
        temp_matrix.append([0.01 * 1.5 * (radius[i]) + 0.01 * 1.5 * (radius[j]) for j in range(len(radius))])
    threshold_df = pd.DataFrame(data=temp_matrix,index=element_list,columns=element_list)
    threshold_df.to_csv(path + "/threshold_matrix_looping.csv")

    work_list = cut_the_work_list(target_group, NP)

    ######################################################################################################################
    ########## SAVE THE GLOBAL FILES FOR LATER CHECK, ACTUALLY SHOULD SAVED A SET OF FILES AS GLOBAL INPUT ###############
    ######################################################################################################################

    pair_info_dict = {}
    for idx_of_list, sub_work_list in enumerate(work_list):
        pool = multiprocessing.Pool(n)
        for i, mid in enumerate(sub_work_list):
            if mid in tolerance_dict:
                t = tolerance_dict[mid]
                abortable_func = partial(abortable_worker_MOT, Global_Spider, timeout=timeout1)
                pool.apply_async(abortable_func, args = (mid,i, t), callback = assemble_length_data)
        pool.close()
        pool.join()

    file_save = open(path + "/global.pkl",'wb') 
    pickle.dump(pair_info_dict, file_save) 
    file_save.close()
    global_pairs_info = global_classify(pair_info_dict)

    global_save_path = path + "/global_length_csv/"
    os.mkdir(global_save_path)
    for k,v in global_pairs_info.items():
        save_bond_length(k, v, global_save_path)


    #####################################################################################################################
    ######################################## THE MAIN PROCESS IN THE LOOP (AT MOST 20 LOOPS) ############################
    #####################################################################################################################
    list_of_MOT = [] #record matrix of threshold.
    list_of_PID = [] #record pairs info list.
    all_result = {}

    loop = 1
    while loop < max_loop:
        pair_info_dict = {}
        result_dict = {}

        for idx_of_list, sub_work_list in enumerate(work_list):
            pool = multiprocessing.Pool(n)
            for i, mid in enumerate(sub_work_list):
                if mid in tolerance_dict:
                    t = tolerance_dict[mid]
                    abortable_func = partial(abortable_worker_MOT, Spider, timeout=timeout1)
                    pool.apply_async(abortable_func, args = (mid,i,t), callback = assemble_length_data)
            pool.close()
            pool.join()

        global_pairs_info = global_classify(pair_info_dict)
        list_of_PID.append(global_pairs_info)

        matrix_of_threshold = np.array(pd.read_csv(path + "/threshold_matrix_looping.csv", header=0, index_col=0))

        pool = multiprocessing.Pool(n)
        for i,ele1 in enumerate(element_list):
            for j,ele2 in enumerate(element_list):
                if i <= j:
                    former_threshold = matrix_of_threshold[element_list.index(ele1)][element_list.index(ele2)]
                    #abortable_func = partial(abortable_worker, get_bond_length_distribution_fitting_info, timeout=1800)
                    if (ele1,ele2) in global_pairs_info or (ele2,ele1) in global_pairs_info:
                        try:
                            refined_data = global_pairs_info[(ele1,ele2)]
                        except:
                            refined_data = global_pairs_info[(ele2,ele1)]
                        pool.apply_async(get_bond_length_distribution_fitting_info, 
                            args = (ele1,ele2,former_threshold,refined_data, global_save_path), callback = assemble_looping_data)
        pool.close()
        pool.join()

        threshold_data = renew_threshold_matrix(result_dict, path)

        list_of_MOT.append(threshold_data)

        #con, osc, all_result = check_convergence(all_result,threshold_data,list_of_PID)
        converged, plotting_info, all_result = check_convergence_new(all_result, threshold_data, loop)

        if converged:
            loop = 100
            time.sleep(60)
        else:
            loop += 1

    time.sleep(60)
    
    file_save = open(path + "/MOT.pkl",'wb') 
    pickle.dump(list_of_MOT, file_save) 
    file_save.close()

    file_save = open(path + "/PID.pkl",'wb') 
    pickle.dump(list_of_PID, file_save) 
    file_save.close()

    file_save = open(path + "/ALLRST.pkl",'wb') 
    pickle.dump(all_result, file_save) 
    file_save.close()

    file_save = open(path + "/plotting_info_dict.pkl",'wb')
    pickle.dump(plotting_info, file_save)
    file_save.close() 

    print("OUT of the LOOP, starting to generate csv and png data!")

    #############################################################################################################
    ############################ SAVE THE VALID CSV AND PNG FILES IN THE PORPER LOOP ############################
    #############################################################################################################
    #plotting_info = get_plotting_info(all_result)

    csv_save_path = path + "/redefined_length_csv/"
    png_save_path = path + "/redefined_length_png/"
    os.mkdir(csv_save_path)
    os.mkdir(png_save_path)

    refined_PID_and_MOT = {}

    for k,v in plotting_info.items():
        try:
            pair_name = (k[0], k[1])
            length_list = list_of_PID[v][pair_name]
            former_threshold = list_of_MOT[v-1][element_list.index(k[0])][element_list.index(k[1])]
            refined_PID_and_MOT[pair_name] = (length_list,former_threshold)
        except:
            pair_name = (k[1], k[0])
            length_list = list_of_PID[v][pair_name]
            former_threshold = list_of_MOT[v-1][element_list.index(k[1])][element_list.index(k[0])]
            refined_PID_and_MOT[pair_name] = (length_list,former_threshold)
        save_bond_length(pair_name, length_list,csv_save_path)

    file_save = open(path + "/refined_PID_and_MOT.pkl",'wb') 
    pickle.dump(refined_PID_and_MOT, file_save) 
    file_save.close()

    target_group = [(k,v) for k,v in refined_PID_and_MOT.items()]
    work_list = cut_the_work_list(target_group, 10)

    fitting_info_dict = {}
    for sub_work_list in work_list:
        pool = Pool(n)
        for work in sub_work_list:
            ele1 = work[0][0]
            ele2 = work[0][1]
            former_threshold = work[1][1]
            refined_data = work[1][0]
            #abortable_func = partial(abortable_worker, bond_length_distribution_fitting, timeout=600)
            pool.apply_async(bond_length_distribution_fitting, 
                args = (ele1,ele2,former_threshold,refined_data,png_save_path,global_save_path), callback = assemble_final_data)
        pool.close()
        pool.join()

    file_save = open(path + "/fitting_info_dict.pkl",'wb')
    pickle.dump(fitting_info_dict, file_save)
    file_save.close() 

    tm = pd.DataFrame(np.zeros((118,118)), columns=element_list, index = element_list)
    for key in [((ele1, ele2),(i,j)) for i,ele1 in enumerate(element_list) for j,ele2 in enumerate(element_list)]:
        if key[0] in fitting_info_dict:
            tm.iloc[key[1][1]][key[1][0]] = fitting_info_dict[key[0]]["threshold"]
            tm.iloc[key[1][0]][key[1][1]] = fitting_info_dict[key[0]]["threshold"]
    tm.to_csv(path + "/threshold_matrix_looped.csv")