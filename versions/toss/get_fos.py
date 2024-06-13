import os
import pandas as pd
import numpy as np
from result import RESULT
from pre_set import PRE_SET
from digest import DIGEST
from get_structure import GET_STRUCTURE
from initialization import INITIAL
from first_algo import FIRST_ALGO
from second_algo import SECOND_ALGO
from resonance import RESONANCE

class GET_FOS():

    def initial_guess(self, m_id, delta_X, tolerance, tolerance_list, res, server=False, filepath="/"):
        if server:
            res = RESULT()
            filepath, m_id = os.path.split(filepath)

        PS = PRE_SET()
        res.dict_ele,\
        res.matrix_of_threshold = PS.dict_ele, PS.matrix_of_threshold 

        res.Forced_transfer_group,\
        res.inorganic_group = PS.Forced_transfer_group, PS.inorganic_group
        #res.transit_metals, res.metals = PS.transit_matals, PS.metals

        if not server:
            GS = GET_STRUCTURE(m_id)
        else:
            GS = GET_STRUCTURE(m_id, specific_path="../toss_server/%s/"%filepath)
        res.sites,\
        res.idx, \
        res.struct = GS.sites, GS.idx, GS.struct

        res.matrix_of_length,\
        res.valence_list, \
        res.elements_list = GS.matrix_of_length, GS.valence_list, GS.elements_list

        benchmark_tolerance, \
        tolerance_list = self.nearest(tolerance, tolerance_list)

        DG = DIGEST(tolerance_list, tolerance, m_id, res)
        res.max_oxi_list, \
        res.min_oxi_list = DG.max_oxi_list, DG.min_oxi_list

        res.SHELL_idx_list, \
        res.threshold_list,\
        res.SHELL_idx_list_with_images = DG.SHELL_idx_list, DG.threshold_list, DG.SHELL_idx_list_with_images

        res.organic_patch, \
        res.alloy_flag = DG.organic_patch, DG.alloy_flag

        DG.digest_structure_with_image(res)
        res.shell_ele_list, \
        res.shell_env_list, \
        res.shell_idx_list, \
        res.shell_CN_list, \
        res.shell_X_list = DG.shell_ele_list, DG.shell_env_list, DG.shell_idx_list, DG.shell_CN_list, DG.shell_X_list

        INT = INITIAL(tolerance, m_id, delta_X, res)
        res.bo_matrix, \
        res.print_covalent_status, \
        res.ori_bo_matrix, \
        res.covalent_pair, \
        res.original_min_oxi_list, \
        res.original_max_oxi_list = INT.bo_matrix, INT.print_covalent_status, INT.ori_bo_matrix, INT.covalent_pair, INT.original_min_oxi_list, INT.original_max_oxi_list 
        
        FA = FIRST_ALGO(m_id, delta_X, tolerance, res)
        res.inorganic_acid_flag, \
        res.first_algorithm_flag = FA.inorganic_acid_flag, FA.first_algorithm_flag

        FA.apply_local_iter_method(res)
        res.inorganic_acid_center_idx = FA.inorganic_acid_center_idx

        FA.apply_first_algo(delta_X,res)
        
        SA = SECOND_ALGO(delta_X, res)
        res.ori_n,\
        res.ori_super_atom_idx_list,\
        res.ori_sum_of_valence = SA.ori_n,SA.ori_super_atom_idx_list,SA.ori_sum_of_valence

        res.sum_of_valence,\
        res.super_atom_idx_list,\
        res.link_atom_idx_list,\
        res.single_atom_idx_list = SA.sum_of_valence,SA.super_atom_idx_list,SA.link_atom_idx_list,SA.single_atom_idx_list

        RSN = RESONANCE(res)
        res.resonance_flag, \
        res.resonance_order = RSN.resonance_flag, RSN.resonance_order   

        if res.resonance_flag:
            res.fake_n, \
            res.exclude_super_atom_list, \
            res.perfect_valence_list, \
            res.plus_n = RSN.fake_n, RSN.exclude_super_atom_list, RSN.perfect_valence_list, RSN.plus_n,

            RSN.erase_bg_charge(res)
        if not server:
            res.species_uni_list = SA.uniformity(res.sum_of_valence, res, res.idx)
        else:
            res.species_uni_list = SA.uniformity(res.sum_of_valence, res, res.idx)
            result = pd.DataFrame(np.vstack([np.array(res.elements_list),np.array(res.sum_of_valence),np.array(res.shell_CN_list)]))
            result.index = ["Elements", "Valence","Coordination Number"]
        return result


    def loss_loop(self, m_id, delta_X, tolerance, tolerance_list, res, server=False, filepath="/"):
        if server:
            res = RESULT()
            filepath, m_id = os.path.split(filepath)

        PS = PRE_SET()
        res.dict_ele,\
        res.matrix_of_threshold = PS.dict_ele, PS.matrix_of_threshold 

        res.Forced_transfer_group,\
        res.inorganic_group = PS.Forced_transfer_group, PS.inorganic_group#PS.inorganic_group  #get rid of the effect of the inorganic group.
        #res.transit_metals, res.metals = PS.transit_matals, PS.metals

        if not server:
            GS = GET_STRUCTURE(m_id)
        else:
            GS = GET_STRUCTURE(m_id, specific_path="../toss_server/%s/"%filepath)
        res.sites,\
        res.idx, \
        res.struct = GS.sites, GS.idx, GS.struct

        res.matrix_of_length, \
        res.valence_list, \
        res.elements_list = GS.matrix_of_length, GS.valence_list, GS.elements_list

        benchmark_tolerance, \
        tolerance_list = self.nearest(tolerance, tolerance_list)

        DG = DIGEST(tolerance_list, tolerance, m_id, res)
        res.max_oxi_list, \
        res.min_oxi_list = DG.max_oxi_list, DG.min_oxi_list

        res.SHELL_idx_list, \
        res.threshold_list = DG.SHELL_idx_list, DG.threshold_list

        res.SHELL_idx_list_with_images = DG.SHELL_idx_list_with_images

        res.organic_patch, \
        res.alloy_flag = DG.organic_patch, DG.alloy_flag

        DG.digest_structure_with_image(res)
        res.shell_ele_list, \
        res.shell_env_list, \
        res.shell_idx_list, \
        res.shell_CN_list, \
        res.shell_X_list = DG.shell_ele_list, DG.shell_env_list, DG.shell_idx_list, DG.shell_CN_list, DG.shell_X_list

        INT = INITIAL(tolerance, m_id, delta_X, res)
        res.bo_matrix, \
        res.print_covalent_status, \
        res.ori_bo_matrix, \
        res.covalent_pair, \
        res.original_min_oxi_list, \
        res.original_max_oxi_list = INT.bo_matrix, INT.print_covalent_status, INT.ori_bo_matrix, INT.covalent_pair, INT.original_min_oxi_list, INT.original_max_oxi_list 

        FA = FIRST_ALGO(m_id, delta_X, tolerance, res)
        res.inorganic_acid_flag,\
        res.first_algorithm_flag = FA.inorganic_acid_flag, FA.first_algorithm_flag

        FA.apply_local_iter_method(res)
        res.inorganic_acid_center_idx = FA.inorganic_acid_center_idx
        FA.apply_first_algo(delta_X,res)

        SA = SECOND_ALGO(delta_X, res)
        res.ori_n,\
        res.ori_super_atom_idx_list,\
        res.ori_sum_of_valence = SA.ori_n,SA.ori_super_atom_idx_list,SA.ori_sum_of_valence

        res.sum_of_valence,\
        res.super_atom_idx_list,\
        res.link_atom_idx_list,\
        res.single_atom_idx_list = SA.sum_of_valence,SA.super_atom_idx_list,SA.link_atom_idx_list,SA.single_atom_idx_list

        RSN = RESONANCE(res)
        res.resonance_flag, \
        res.resonance_order = RSN.resonance_flag, RSN.resonance_order       

        res.species_uni_list = SA.uniformity(res.sum_of_valence, res, res.super_atom_idx_list)
        if server:
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
"""END HERE"""