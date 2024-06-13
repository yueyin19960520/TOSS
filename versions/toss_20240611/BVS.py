import time
import pickle
import os
import re
import copy
import random
import pandas as pd
import scipy.sparse as sp
import numpy as np
import argparse

from result import RESULT
from get_fos import GET_FOS
from tune import TUNE
from post_process import *
from auxilary import *

from multiprocessing.dummy import Pool as ThreadPool
from functools import partial 
import multiprocessing

from pymatgen.analysis.bond_valence import BVAnalyzer 
from pymatgen.core.structure import IStructure


def cal_BVS(i, mid, path):
    OS = BVAnalyzer()
    try:
        struct = IStructure.from_file(path + mid)
        print("Get the %s-th structre of the name %s."%(i, mid))
        valence_list = OS.get_valences(struct)
        print("Get the %s-th valences of the name %s successfully."%(i, mid))
    except Exception as e:
        print(f"An error occurred: {e}")
        ele_list = list(map(lambda x:x.specie.name, struct.sites))
        if len(set(ele_list)) == 1:
            valence_list = [0] * len(ele_list)
        else:
            valence_list = None

    parameters = [mid, valence_list]
    return parameters


def assemble_BVS(parameters):
    global valence_dict
    global failed_structure
    global stuked_structure

    if parameters[1] != None:
        valence_dict.update({parameters[0]: parameters[1]})
    else:
        failed_structure.append(parameters[0])

    finished_len = len(valence_dict) + len(failed_structure) + len(stuked_structure)
    print("Successed_Structure_Number:%s, Failed_Structure_Number:%s, Stucked_Structure_Number:%s"%(len(valence_dict),len(failed_structure),len(stuked_structure)))
    print("The process has finished %s/%s."%(finished_len, len(target_group)))


def assemble_TOSS(parameters):
    global my_valence_dict
    global failed_structure
    global stuked_structure
    global graphs_dict

    if parameters[1] != []:
        my_valence_dict.update({parameters[0]: [parameters[1],parameters[2]]})
        graphs_dict.update({parameters[0]:parameters[-1]})
    else:
        if parameters[2] == "RERUN":
            stuked_structure.append(parameters[0])
        else:
            failed_structure.append(parameters[0])

    finished_len = len(my_valence_dict) + len(failed_structure) + len(stuked_structure)
    print("Successed_Structure_Number:%s, Failed_Structure_Number:%s, Stucked_Structure_Number:%s"%(len(my_valence_dict),len(failed_structure),len(stuked_structure)))
    print("The process has finished %s/%s."%(finished_len, len(target_group)))


def sort_valence_by_elements(valence_list, element_list):
    v_e = [(i,j) for i,j in zip(valence_list, element_list)]
    sort_v_e = sorted(v_e, key = lambda x:[x[0],x[1]])
    sort_v = [ve[0] for ve in sort_v_e]
    return sort_v


def cal_TOSS_init(i, mid):
    tolerance_list = valid_t_dict[mid]

    result_candi = []
    alloy_flag = None
    for t in tolerance_list:
        try:
            res = RESULT()
            GFOS = GET_FOS()
            GFOS.initial_guess(m_id = mid, delta_X = 0.1, tolerance = t, tolerance_list = tolerance_list, res = res)

            class_with_idx = {}
            for I in res.idx:
                key = tuple([res.elements_list[I], tuple(sorted(res.shell_ele_list[I]))])
                if key in class_with_idx:
                    class_with_idx[key].append(I)
                else:
                    class_with_idx[key] = [I]

            sort_valence_list = sort_valence_by_elements(res.sum_of_valence, res.elements_list)
            result_candi.append((sort_valence_list, res.min_oxi_list, res.max_oxi_list, res.elements_list, class_with_idx, res.sum_of_valence))
            alloy_flag = res.alloy_flag

        except Exception as e:
            print(f"An error occurred: {e}")

    print("Get the %s-th valences of the name %s successfully."%(i, mid)) if result_candi != [] else print("Failed to get the %s-th structre of the name %s."%(i,mid))
    parameters = [mid,result_candi, alloy_flag, None]
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    return parameters


def cal_TOSS_loop(i, m_id):
    GFOS = GET_FOS()
    delta_X = 0.1
    tolerance_list = valid_t_dict[m_id]
    corr_t = []
        
    for t in tolerance_list:
        res = RESULT()
        TN = TUNE()
        try:
            GFOS.loss_loop(m_id, delta_X, t, tolerance_list, res)
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
            
            coef = 1.2
            loss_value = coef**N_spec * LOSS
            corr_t.append((t,loss_value,res))
        except Exception as e:
            print(f"An error occurred: {e}")

    #print(len(corr_t))
    result_candi = []
    alloy_flag = None
    G = None
    if len(corr_t) > 0:
        chosen_one = sorted(corr_t, key = lambda x:x[1])[0]
        res = chosen_one[2]

        class_with_idx = {}
        for I in res.idx:
            key = tuple([res.elements_list[I], tuple(sorted(res.shell_ele_list[I]))])
            if key in class_with_idx:
                class_with_idx[key].append(I)
            else:
                class_with_idx[key] = [I]

        sort_valence_list = sort_valence_by_elements(res.final_vl, res.elements_list)
        result_candi.append([sort_valence_list, res.min_oxi_list, res.max_oxi_list, res.elements_list, class_with_idx, res.final_vl])
        alloy_flag = res.alloy_flag

        ###############################################################################################################################
        index_list = np.array(res.idx, dtype="float32")
        elements_list = res.elements_list
        elements_list_one_hot = np.array([res.periodic_table.elements_list.index(e)+1 for e in res.elements_list],dtype="float32")
        CN_list = np.array(res.shell_CN_list,dtype="float32")
        SEN_list = np.array(res.shell_env_list,dtype="float32")
        IP_matrix = np.array([np.array([ip for ip in res.dict_ele[e]["IP"][0:8]], dtype="float32") for e in elements_list]).T
        EN_list = np.array([res.dict_ele[e]["X"] for e in elements_list],dtype="float32")
        OS_list = np.array([os for os in res.sum_of_valence],dtype="float32")
        R1_list = np.array([res.dict_ele[e]["covalent_radius"] for e in elements_list], dtype="float32")
        R2_list = np.array([res.dict_ele[e]["second_covalent_radius"] for e in elements_list], dtype="float32")
        R3_list = np.array([res.dict_ele[e]["third_covalent_radius"] for e in elements_list], dtype="float32")

        node_feats = np.vstack([elements_list_one_hot,EN_list,CN_list,SEN_list,R1_list,R2_list,R3_list,IP_matrix,OS_list]).T

        #generate edge features:
        edge_feats = []
        for I,bo_list in enumerate(res.ori_bo_matrix):
            for J, bo in enumerate(bo_list):
                valid_length = res.matrix_of_length[I][J] if bo != 0 else 0 
                if valid_length != 0:
                    edge_feats.append([I, J , valid_length])
        #edge_feats = np.vstack(distances)

        edge_df = pd.DataFrame(edge_feats, columns=["Src", "Dst", "Length"])
        edge_df["Src"] = edge_df["Src"].astype("int")
        edge_df["Dst"] = edge_df["Dst"].astype("int")
        edge_df["Length"] = edge_df["Length"].astype("float64")
        node_df = pd.DataFrame(node_feats, columns=["Element","EN","CN","SEN","R1","R2","R3","IP1","IP2","IP3","IP4","IP5","IP6","IP7","IP8","OS"])
        G = {"n": node_df, "e":edge_df}
        ###############################################################################################################################

    else:
        None

    parameters = [m_id, result_candi, alloy_flag, G]
    if result_candi != []:
        print("Get the %s-th valences of the name %s successfully."%(i, m_id))
    else:
        print("Failed to get the %s-th structre of the name %s."%(i,m_id))
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    return parameters


def abortable_worker(func, *args, **kwargs):
    timeout = kwargs.get('timeout', None)
    p = ThreadPool(1)
    res = p.apply_async(func, args = args)
    try:
        out = res.get(timeout)
    except multiprocessing.TimeoutError:
        p.terminate()
        out = [args[1], [], "RERUN"]
    finally:
        p.close()
        p.join()
    return out


def success_rate(valence_dict, my_valence_dict):
    alloy = []  #both right, the alloy structures
    break_match = []  #which break the symmetry criteria 
    exceed = []  #exceed the lower or upper valence limit
    seperation = []  #the seperation of link atom and the superatom
    ORE = [] #the degree of the oxidation and reduction
    possible_tune = [] #it could be tuned by tunation

    s = 0 #successful number
    f = 0 #break match number (smaller thoretically)
    F = 0 #break match number (larger thoretically)
    a = 0 #alloy number
    b = 0 #exceed the bound number (smaller thoretically)
    B = 0 #exceed the bound number (larger thoretically)
    p = 0 #classification is right, but the sum of electrons in each group is not same
    P = 0 #classification is wrong
    w = 0 #why we are different

    for k,V in my_valence_dict.items(): #V == [result_candi, alloy_flag]
        v = V[0]
        alloy_flag = V[1]
        if alloy_flag == True:
            a += 1
            #Record the structure with the TRUE alloy flag.
            alloy.append(k)
        else:
            my_valence_list = [i[0] for i in v]   #v == [[vl, min, max, ele, classes],[vl, min, max, ele, classes]]
            if len(my_valence_list) > 0: 
                BVS_vl = valence_dict[k]
                ele_list = v[0][3]
                sort_BVS_vl = sort_valence_by_elements(BVS_vl, ele_list)
                if sort_BVS_vl in my_valence_list:
                    s += 1
    #######################################################################################################################
                #Record the structures that break the symmatry criterion.
                else:
                    for candi in v:  #candi == [vl, min, max, ele, classes]
                        classes = candi[4]
                        for kk,vv in classes.items():
                            check_valence_list = [BVS_vl[ii] for ii in vv]
                            if len(set(check_valence_list)) != 1:
                                F += 1
                                break_match.append(k)
                                break
                        break
        
                    sub_f = 0
                    for candi in v:
                        for kk,vv in classes.items():
                            check_valence_list = [BVS_vl[ii] for ii in vv]
                            if len(set(check_valence_list)) != 1:
                                sub_f += 1
                                break
                    if sub_f == len(v):
                        f += 1
    ########################################################################################################################
                    #Record the structures that the valences exceed the lower or upper bounnd.
                    if k not in break_match: #do not break the symmetry criterion
                        for candi in v:
                            min_oxi_list = candi[1]
                            max_oxi_list = candi[2]
                            for i,val in enumerate(BVS_vl):
                                upper_bound = max_oxi_list[i]
                                lower_bound = min_oxi_list[i]
                                if val < lower_bound or val > upper_bound:
                                    B += 1
                                    exceed.append(k)
                                    break
                            break

                        sub_b = 0
                        for candi in v:
                            min_oxi_list = candi[1]
                            max_oxi_list = candi[2]
                            for i,val in enumerate(BVS_vl):
                                upper_bound = max_oxi_list[i]
                                lower_bound = min_oxi_list[i]
                                if val < lower_bound or val > upper_bound:
                                    sub_b += 1
                                    break 
                        if sub_b == len(v):
                            b += 1

    ##########################################################################################################################
                        if k not in exceed:
                            BVS_minus = []
                            BVS_plus = []
                            BVS_minus_val = []
                            for ii, val in enumerate(BVS_vl):
                                if val < 0:
                                    BVS_minus.append(ii)
                                    BVS_minus_val.append(val)
                                else:
                                    BVS_plus.append(ii)

                            for candi in v:
                                valence_list = candi[5]
                                my_minus = []
                                my_plus = [] #0 also consider in here
                                my_minus_val = []
                                for ii, val in enumerate(valence_list):
                                    if val < 0:
                                        my_minus.append(ii)
                                        my_minus_val.append(val)
                                    else:
                                        my_plus.append(ii)

                                if set(my_minus) != set(BVS_minus):
                                    seperation.append(k)
                                    P += 1
                                else:
                                    if sum(my_minus_val) != sum(BVS_minus_val):
                                        ORE.append(k)
                                        p += 1
                                        break
                                    else:
                                        None
                                break

                            if k not in seperation+ORE:
                                #make sure not break the symmetry criterion
                                for candi in v:
                                    check_classes = candi[4]

                                    classified_valence_list = [[BVS_vl[ii] for ii in sub_idx] for sub_idx in check_classes.values()]
                                    set_valence_list = [list(set(l)) for l in classified_valence_list]
                                    minus_one = [len(l) for l in set_valence_list if (np.array(l) < 0).all()]
                                    plus_one =  [len(l) for l in set_valence_list if (np.array(l) >= 0).all()]
                                    why_one = [len(l) for l in set_valence_list if (np.array(l) < 0).any() and (np.array(l) >= 0).any()]

                                    if set(minus_one) == {1} and set(plus_one) == {1} and len(why_one) == 0:
                                        #pass the exam!
                                        my_vl = candi[5]
                                        my_sum_minus = sum([vvv for vvv in my_vl if vvv < 0])
                                        my_sum_plus = sum([vvv for vvv in my_vl if vvv >= 0])

                                        BVS_sum_minus = sum([vvv for vvv in BVS_vl if vvv < 0])
                                        BVS_sum_plus = sum([vvv for vvv in BVS_vl if vvv >= 0])

                                        if my_sum_plus == BVS_sum_plus and my_sum_minus == my_sum_minus:
                                            #pass the exam!
                                            possible_tune.append(k)
                                            w += 1 
                                            break
                                        else:
                                            None
                                    else:
                                        None
            else:
                None
    return s,f,F,a,B,b,P,p,w, alloy, break_match, exceed, seperation,ORE,possible_tune


##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
path = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]

valence_dict = {}
failed_structure = []
stuked_structure = []

#target_group = os.listdir(path + "/structures")
target_group = os.listdir("D:/share/TOSS_2024/structures/")#random.sample(target_group, 10000)

file_get = open(path + "/valid_t_dict.pkl",'rb')
valid_t_dict = pickle.load(file_get)
file_get.close() 

structure_path = "D:/share/TOSS_2024/structures/"#path + "/structures/" ##
suffix  = "loop_" + str(max(list(map(int,list(map(lambda x:re.findall(".*loop_(.*).pkl(.*)",x)[0][0], list(filter(lambda x:"loop_" in x,os.listdir(path)))))))))


file_get= open(path + "/global_normalized_normed_dict_%s.pkl"%suffix,"rb")
global_normalized_normed_dict = pickle.load(file_get)
file_get.close()
#matched_dict = global_normalized_normed_dict

file_get= open(path + "/global_mean_dict_%s.pkl"%suffix,"rb")
global_mean_dict = pickle.load(file_get)
file_get.close()

file_get= open(path + "/global_sigma_dict_%s.pkl"%suffix,"rb")
global_sigma_dict = pickle.load(file_get)
file_get.close()
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--ncore', type=int, required=True, help='Number of Parallel')
    parser.add_argument('-t1', '--timeout1', type=int, default=3000, help="The primary timeout seconds for each subprocess.")
    parser.add_argument('-t2', '--timeout2', type=int, default=3000, help="The secondary timeout seconds for each subprocess.")
    args = parser.parse_args()

    n = args.ncore
    primary_limit_time = args.timeout1
    secondary_limit_time = args.timeout2

    NP = 600000
    work_list = cut_the_work_list(target_group, NP) 
    print("This run contains %d structures."%len(target_group))
    print("Main is processing......")

    for idx_of_list, sub_work_list in enumerate(work_list):
        pool = multiprocessing.Pool(n)
        for i, mid in enumerate(sub_work_list):
            pool.apply_async(cal_BVS, args = (i,mid,structure_path,), callback = assemble_BVS)
        pool.close()
        pool.join()

    file_save = open(path + "/BVS_result.pkl",'wb') 
    pickle.dump(valence_dict, file_save) 
    file_save.close()

    print("FINISHED THE BVS RESULT CALCULATION!")

    #############################################################################################################
    ####################################TOSSTOSSTOSSTOSSTOSSTOSS#################################################
    #############################################################################################################

    target_group = list(valence_dict.keys())
    work_list = cut_the_work_list(target_group, NP) 

    for inner_func in ["TUNE"]:
        my_valence_dict = {}
        failed_structure = []
        stuked_structure = []
        graphs_dict = {}

        if inner_func == "TUNE":
            for idx_of_list, sub_work_list in enumerate(work_list):
                pool = multiprocessing.Pool(n)
                for i, mid in enumerate(sub_work_list):
                    abortable_func = partial(abortable_worker, cal_TOSS_loop, timeout=primary_limit_time)
                    pool.apply_async(abortable_func, args = (i,mid,), callback = assemble_TOSS)
                pool.close()
                pool.join()
        else:
            for idx_of_list, sub_work_list in enumerate(work_list):
                
                pool = multiprocessing.Pool(n)
                for i, mid in enumerate(sub_work_list):
                    abortable_func = partial(abortable_worker, cal_TOSS_init, timeout=secondary_limit_time)
                    pool.apply_async(abortable_func, args = (i,mid,), callback = assemble_TOSS)
                pool.close()
                pool.join()


        print("FINISHED THE YUE RESULT CALCULATION!")

        s,f,F,a,B,b,P,p,w, alloy, break_match, exceed, seperation,ORE,possible_tune = success_rate(valence_dict, my_valence_dict)
        A = s + F + a + B + P + p + w

        with open(path + "/BVS_result.txt", "a+") as f:
            f.write("%s:\n"%inner_func)
            f.write("The BVS got %s results.\n"%len(valence_dict))
            f.write("This program got %s results.\n"%len(my_valence_dict))
            f.write("The have %s structures' results matched!\n"%s)
            f.write("%s BVS structures's result break the criterion of symmetry!\n"%F)
            f.write("%s structures are alloy structures.\n"%a)
            f.write("%s BVS structures's result exceed the upper or lower bound of the oxidation state!\n"%B)
            f.write("%s seperations of the link-atom and super-atom are different!\n"%P)
            f.write("%s oxidation and reduction degree are different!\n"%p)
            f.write("%s structures's result could be tuned by the loss function.\n"%w)
            f.write("%s(%s) sturctures have been analysed the result.\n"%(A,round((A*100/len(my_valence_dict)),2)))
            f.close()

        try:
            sent_message(value3 = "%s~%s~%s success rate!"%(round((s)/len(valence_dict) * 100, 2),
                                                            round((A)/len(valence_dict) * 100, 2), 
                                                            round((A)/len(my_valence_dict) * 100, 2)))
        except:
            None


        file_save = open(path + "/YUE_result_%s.pkl"%inner_func,'wb') 
        pickle.dump(my_valence_dict, file_save) 
        file_save.close()

        dismatch = {"ALLOY":alloy,
                    "break_symmetry":break_match,
                    "exceed_limit":exceed,
                    "wrong_seperation":seperation,
                    "OR_degree":ORE,
                    "tunation_candi":possible_tune}

        file_save = open(path + "/dismatch_%s.pkl"%inner_func,'wb') 
        pickle.dump(dismatch, file_save) 
        file_save.close()

    # Save a graph_dict for all the valid data.
    file_save = open(path + "/TOSS_graphs_dict.pkl",'wb')
    pickle.dump(graphs_dict, file_save) 
    file_save.close()
    TOSS_graphs_dict = copy.deepcopy(graphs_dict)


    # Find the graphs that BVS and TOSS give the same result.
    cool = list(my_valence_dict.keys())
    hot = [j for i in [v for k,v in dismatch.items()] for j in i]
    target_group = list(set(cool).difference(set(hot)))

    none_value = set(graphs_dict.keys()).difference(set(target_group))

    for mid in none_value:
        del(graphs_dict[mid])

    # Save the double check graphs_dict but without alloys.
    file_save = open(path + "/graphs_dict.pkl",'wb')
    pickle.dump(graphs_dict, file_save) 
    file_save.close()

    """
    # Adding some alloys for better transferbility.
    alloy_ids = random.sample(alloy, 20000)
    alloy_sample_graphs_dict = {mid:TOSS_graphs_dict[mid] for mid in alloy_ids}
    graphs_dict.update(alloy_sample_graphs_dict)

    file_save = open(path + "/graphs_dict.pkl",'wb')
    pickle.dump(graphs_dict, file_save) 
    file_save.close()
    """


"""END HERE"""