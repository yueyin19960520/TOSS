from get_fos import GET_FOS
from post_process import *
import time
import pickle
import os
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial 
import multiprocessing
import copy
from auxilary import *
import pandas as pd

from result import RESULT
from pre_set import PRE_SET
from digest import DIGEST
from get_structure import GET_STRUCTURE
from initialization import INITIAL
from first_algo import FIRST_ALGO
from second_algo import SECOND_ALGO
from resonance import RESONANCE
from tune import TUNE
from visualization import VS
import random
import openpyxl


#structure_path = "D:/share/old_TOSS/structures/"


class Development():
    def __init__(self, structure_path, global_normalized_normed_dict, global_mean_dict, global_sigma_dict):
        self.structure_path = structure_path
        self.global_normalized_normed_dict = global_normalized_normed_dict
        self.global_mean_dict = global_mean_dict
        self.global_sigma_dict = global_sigma_dict


    def get_CN_list(self, m_id, t, valid_t):

        # Generate the coordination environment of a given structru

        res = RESULT()
        PS = PRE_SET(spider = False)
        res.dict_ele, res.matrix_of_threshold = PS.dict_ele, PS.matrix_of_threshold
        res.Forced_transfer_group,res.inorganic_group = PS.Forced_transfer_group,PS.inorganic_group

        GS = GET_STRUCTURE(m_id, self.structure_path)############################## DEFINE THE STRUCTURE PATH HERE##########################
        res.sites,res.idx, res.struct = GS.sites, GS.idx, GS.struct
        res.matrix_of_length, res.valence_list, res.elements_list = GS.matrix_of_length, GS.valence_list, GS.elements_list

        DG = DIGEST(valid_t, t, m_id, res)
        res.max_oxi_list, res.min_oxi_list = DG.max_oxi_list, DG.min_oxi_list
        res.SHELL_idx_list, res.threshold_list = DG.SHELL_idx_list, DG.threshold_list
        res.organic_patch, res.alloy_flag = DG.organic_patch, DG.alloy_flag
        DG.digest_structure_with_image(res)
        res.shell_ele_list, res.shell_env_list, res.shell_idx_list, res.shell_CN_list, res.shell_X_list = DG.shell_ele_list, DG.shell_env_list, DG.shell_idx_list, DG.shell_CN_list, DG.shell_X_list
        return res



    def get_the_valid_t(self, m_id,i):

        # USED FOR GETTING TOLERANCE LIST

        # valid_t_list = get_the_valid_t(m_id,i=0)
        # print(valid_t_list)
     
        # res = get_CN_list(m_id, 1.1,valid_t_list)
        # res.shell_ele_list

        res = RESULT()
        PS = PRE_SET(spider = False)
        res.dict_ele, res.matrix_of_threshold = PS.dict_ele, PS.matrix_of_threshold
        res.Forced_transfer_group,res.inorganic_group = PS.Forced_transfer_group,PS.inorganic_group

        GS = GET_STRUCTURE(m_id, self.structure_path)  ############################## DEFINE THE STRUCTURE PATH HERE##########################
        res.sites,res.idx, res.struct = GS.sites, GS.idx, GS.struct
        res.matrix_of_length, res.valence_list, res.elements_list = GS.matrix_of_length, GS.valence_list, GS.elements_list

        valid_t = []
        check_result = []

        for t in [round(1.1 + 0.01 * i,2) for i in range(16)]:
            #if True:
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
            except:
            #else:
                None

        print('This is the %sth structure with mid %s and we got %s different valid tolerance(s).'%(i, m_id, len(valid_t)))
        return valid_t



    def initial_guess(self, m_id, delta_X, tolerance, tolerance_list):

        # USED FOR INITIAL GUESS RESULT

        # valid_t_list = get_the_valid_t(m_id,i=0)
        # print(valid_t_list)
        # res = initial_guess(m_id, delta_X=0.1, tolerance=1.1, tolerance_list=valid_t_list)
        # result = pd.DataFrame(np.vstack([np.array(res.elements_list),np.array(res.sum_of_valence),np.array(res.shell_CN_list)]))
        # result.index = ["Elements", "Valence","Coordination Number"]
        # pd.set_option("display.max_columns", 1000)
        # result

        res = RESULT()
        PS = PRE_SET()
        res.dict_ele,res.matrix_of_threshold = PS.dict_ele, PS.matrix_of_threshold 

        res.Forced_transfer_group,\
        res.inorganic_group = PS.Forced_transfer_group, PS.inorganic_group
        #res.transit_metals, res.metals = PS.transit_matals, PS.metals

        GS = GET_STRUCTURE(m_id, self.structure_path)############################## DEFINE THE STRUCTURE PATH HERE##########################
        res.sites,res.idx,res.struct = GS.sites, GS.idx, GS.struct

        res.matrix_of_length,res.valence_list, res.elements_list = GS.matrix_of_length, GS.valence_list, GS.elements_list

        benchmark_tolerance, tolerance_list = self.nearest(tolerance, tolerance_list)

        DG = DIGEST(tolerance_list, tolerance, m_id, res)
        res.max_oxi_list, res.min_oxi_list = DG.max_oxi_list, DG.min_oxi_list

        res.SHELL_idx_list,res.threshold_list,res.SHELL_idx_list_with_images = DG.SHELL_idx_list, DG.threshold_list, DG.SHELL_idx_list_with_images

        res.organic_patch, res.alloy_flag = DG.organic_patch, DG.alloy_flag

        DG.digest_structure_with_image(res)
        res.shell_ele_list, res.shell_env_list, res.shell_idx_list, res.shell_CN_list, res.shell_X_list = DG.shell_ele_list, DG.shell_env_list, DG.shell_idx_list, DG.shell_CN_list, DG.shell_X_list

        INT = INITIAL(tolerance, m_id, delta_X, res)
        res.bo_matrix, res.print_covalent_status, res.ori_bo_matrix, res.covalent_pair = INT.bo_matrix, INT.print_covalent_status, INT.ori_bo_matrix, INT.covalent_pair

        FA = FIRST_ALGO(m_id, delta_X, tolerance, res)
        res.inorganic_acid_flag, res.first_algorithm_flag = FA.inorganic_acid_flag, FA.first_algorithm_flag

        FA.apply_local_iter_method(res)
        res.inorganic_acid_center_idx = FA.inorganic_acid_center_idx

        FA.apply_first_algo(delta_X,res)

        SA = SECOND_ALGO(delta_X, res)
        res.ori_n,res.ori_super_atom_idx_list,res.ori_sum_of_valence = SA.ori_n,SA.ori_super_atom_idx_list,SA.ori_sum_of_valence

        res.sum_of_valence,res.super_atom_idx_list,res.link_atom_idx_list,res.single_atom_idx_list = SA.sum_of_valence,SA.super_atom_idx_list,SA.link_atom_idx_list,SA.single_atom_idx_list    
            
        RSN = RESONANCE(res)
        res.resonance_flag, res.resonance_order = RSN.resonance_flag, RSN.resonance_order

        if res.resonance_flag:
            res.fake_n, res.exclude_super_atom_list, res.perfect_valence_list, res.plus_n = RSN.fake_n, RSN.exclude_super_atom_list, RSN.perfect_valence_list, RSN.plus_n,

            RSN.erase_bg_charge(res)
        res.species_uni_list = SA.uniformity(res.sum_of_valence, res, res.idx)
        return res



    def nearest(self, tolerance, tolerance_list):
        if tolerance_list[-1] <= tolerance:
            benchmark_t = tolerance_list[-1]
        elif tolerance_list[0] >= tolerance:
            benchmark_t = tolerance_list[0]
        else:
            for i,t in enumerate(tolerance_list):
                if t > tolerance:
                    break
            benchmark_t = tolerance_list[i-1]
        refined_tolerance_list = [t for t in tolerance_list if t <= tolerance]
        return benchmark_t, refined_tolerance_list



    def development_loss_loop(self, m_id, delta_X, tolerance, tolerance_list, res):
        PS = PRE_SET()
        res.dict_ele,res.matrix_of_threshold = PS.dict_ele, PS.matrix_of_threshold 

        res.Forced_transfer_group,res.inorganic_group = PS.Forced_transfer_group, PS.inorganic_group
        GS = GET_STRUCTURE(m_id, self.structure_path)
        res.sites,res.idx, res.struct = GS.sites, GS.idx, GS.struct

        res.matrix_of_length, res.valence_list, res.elements_list = GS.matrix_of_length, GS.valence_list, GS.elements_list
        
        benchmark_tolerance, tolerance_list = self.nearest(tolerance, tolerance_list)

        DG = DIGEST(tolerance_list, tolerance, m_id, res)
        res.max_oxi_list, res.min_oxi_list = DG.max_oxi_list, DG.min_oxi_list

        res.SHELL_idx_list, res.threshold_list = DG.SHELL_idx_list, DG.threshold_list

        res.SHELL_idx_list_with_images = DG.SHELL_idx_list_with_images

        res.organic_patch, res.alloy_flag = DG.organic_patch, DG.alloy_flag

        DG.digest_structure_with_image(res)
        res.shell_ele_list, res.shell_env_list, res.shell_idx_list, res.shell_CN_list, res.shell_X_list = DG.shell_ele_list, DG.shell_env_list, DG.shell_idx_list, DG.shell_CN_list, DG.shell_X_list

        INT = INITIAL(tolerance, m_id, delta_X, res)
        res.bo_matrix, res.print_covalent_status, res.ori_bo_matrix,res.covalent_pair = INT.bo_matrix, INT.print_covalent_status, INT.ori_bo_matrix, INT.covalent_pair

        FA = FIRST_ALGO(m_id, delta_X, tolerance, res)
        res.inorganic_acid_flag, res.first_algorithm_flag = FA.inorganic_acid_flag, FA.first_algorithm_flag

        FA.apply_local_iter_method(res)
        res.inorganic_acid_center_idx = FA.inorganic_acid_center_idx
        FA.apply_first_algo(delta_X,res)

        SA = SECOND_ALGO(delta_X, res)
        res.ori_n, res.ori_super_atom_idx_list, res.ori_sum_of_valence = SA.ori_n,SA.ori_super_atom_idx_list,SA.ori_sum_of_valence

        res.sum_of_valence, res.super_atom_idx_list, res.link_atom_idx_list, res.single_atom_idx_list = SA.sum_of_valence,SA.super_atom_idx_list,SA.link_atom_idx_list,SA.single_atom_idx_list

        RSN = RESONANCE(res)
        res.resonance_flag, res.resonance_order = RSN.resonance_flag, RSN.resonance_order

        res.species_uni_list = SA.uniformity(res.sum_of_valence, res, res.super_atom_idx_list)
        return res



    def cal(self, m_id, atom_pool, tolerance_list):

        # USED FOR LOSS LOOP RESULT
        # parameters,res = cal(m_id, i=0, atom_pool="all", tolerance_list=valid_t_list)

        GFOS = GET_FOS()
        delta_X = 0.1
        corr_t = []
        ls = time.time()
            
        for t in tolerance_list:
            res = RESULT()
            TN = TUNE()
            try:
            #if True:
                res = self.development_loss_loop(m_id, delta_X, t, tolerance_list, res)
                temp_pair_info = spider_pair_length_with_CN_unnorm(res.sum_of_valence, res)

                #now, the matched dict is the global normalization normed dict. 
                loss = cal_loss_func_by_MAP(temp_pair_info, 
                                            self.global_normalized_normed_dict, 
                                            self.global_sigma_dict, 
                                            self.global_mean_dict)
                N_spec = len(res.species_uni_list)
                res.initial_vl = res.sum_of_valence
                
                if len(res.super_atom_idx_list) > 0:
                    if res.resonance_flag:
                        avg_LOSS, the_resonance_result = TN.tune_by_resonance(loss,
                                                                              res, 
                                                                              self.global_normalized_normed_dict,
                                                                              self.global_sigma_dict, 
                                                                              self.global_mean_dict)
                        res.final_vl = the_resonance_result[0][0]
                        same_after_resonance = True if res.final_vl == res.initial_vl else False
                        res.sum_of_valence = res.final_vl

                    if atom_pool == "super":
                        process_atom_idx_list = res.super_atom_idx_list
                    if atom_pool == "link":
                        process_atom_idx_list = res.link_atom_idx_list
                    if atom_pool == "all":
                        process_atom_idx_list = res.idx 

                    LOSS, res.final_vl = TN.tune_by_redox_in_certain_range_by_MAP(process_atom_idx_list, 
                                                                                  loss, 
                                                                                  res.sum_of_valence,
                                                                                  0,
                                                                                  res,
                                                                                  self.global_normalized_normed_dict,
                                                                                  self.global_sigma_dict, 
                                                                                  self.global_mean_dict)
                    res.sum_of_valence = res.final_vl
                    same_after_tunation = True if res.final_vl == res.initial_vl else False
                    same_after_resonance = True
                    
                else:
                    res.final_vl = res.initial_vl
                    same_after_tunation = True
                    same_after_resonance = True
                    LOSS = loss
                    
                parameters = {m_id: [res.resonance_flag, same_after_tunation, same_after_resonance]}
                
                loss_value = 1**N_spec * LOSS
                corr_t.append((t,loss_value,res))
            except:
            #else:
                None

        try:
        #if True:
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
            print("Got the Formal Oxidation State of the structure %s in %s seconds."%(m_id,tc))
            print("There one total %s valid tolerance(s), and the most 'proper' one is %s for this structure."%(len(tolerance_list),t))
            for info in corr_t:
                print("Tolerace: %s, Loss: %s"%(info[0],info[1]))
        except:
        #else:
            parameters = [m_id, None, None, None]
            tc = time.time() - ls
            print("Failed to analyze the %sth structure %s in %s seconds."%(i,m_id,tc))
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        return parameters,res
"""END HERE"""