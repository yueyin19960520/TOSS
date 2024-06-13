from polyhedron_algo import *
from digest import get_ele_from_sites
from digest import get_propoteries_of_atom
from post_process import *
import copy


def fake_the_structure(res):
    fake_n = 0
    plus_n = 0
    exclude_super_atom_list = []
    perfect_valence_list = []
    alkali = res.periodic_table.alkali
    earth_alkali = res.periodic_table.earth_alkali
    set_value_ele_dict = {}
    set_value_ele_dict.update(alkali)
    set_value_ele_dict.update(earth_alkali)

    for i in res.super_atom_idx_list:
        ele = get_ele_from_sites(i,res)
        if ele in set_value_ele_dict:
            exclude_super_atom_list.append(i)
            perfect_valence_list.append((i, set_value_ele_dict[ele]))
            if res.sum_of_valence[i] != set_value_ele_dict[ele]:
                fake_n += (set_value_ele_dict[ele] - res.sum_of_valence[i])

    resonance_order = "1"
    for i in exclude_super_atom_list:
        ele = get_ele_from_sites(i,res)
        if ele in earth_alkali:
            resonance_order = "2"
        plus_n += (set_value_ele_dict[ele] - res.ori_sum_of_valence[i])
        #erase the case that the earth_alkali elements are undersaturation
        #for example, Ca-O, the Ca will be +1. 
    return fake_n, exclude_super_atom_list, perfect_valence_list, plus_n, resonance_order


def put_fake_charge_back_to_super_atom(N, atom_idx_list, vl, res):
    while N > 0:
        atom_idx_list = [i for i in atom_idx_list if vl[i] > 0]
        now_ip_list = [res.dict_ele[get_ele_from_sites(i,res)]["IP"][vl[i]-1] for i in atom_idx_list]
        now_env_list = [res.shell_env_list[j] for j in atom_idx_list]
        idx_ip_env = [(i,j,-k) for i,j,k in zip(atom_idx_list, now_ip_list, now_env_list)]
        sorted_idx_ip_env = sorted(idx_ip_env, reverse = True, key = lambda x:[x[1],x[2]])
        op_idx = sorted_idx_ip_env[0][0]
        vl[op_idx] -= 1
        N -= 1
        # Here we can jump the valid OS list, because the resonance valence can be unnormal.
    return vl


def put_fake_charge_back_to_link_atom(N, atom_idx_list, vl, res):
    while N > 0:
        now_v_list = [vl[i] for i in atom_idx_list]
        now_X_list = [res.dict_ele[get_ele_from_sites(i,res)]["X"] for i in atom_idx_list]
        now_env_list = [res.shell_env_list[i] for i in atom_idx_list]
        idx_v_env_X = [(i,(j+l),k) for i,j,k,l in zip(atom_idx_list, now_v_list, now_env_list, now_X_list)]
        sorted_idx_v_env_X = sorted(idx_v_env_X, reverse = True, key = lambda x:[x[1],x[2]])
        op_idx = sorted_idx_v_env_X[0][0]
        vl[op_idx] += 1
        N -= 1
        # Same reason like before.
    return vl


class RESONANCE():
	def __init__(self,res):
		if not res.alloy_flag:
			self.fake_n, self.exclude_super_atom_list, self.perfect_valence_list, self.plus_n, self.resonance_order = fake_the_structure(res)
		else:
			self.fake_n, self.resonance_order = 0, None
		self.resonance_flag = True if self.fake_n != 0 else False


	def erase_bg_charge(self, res):
		valence_A = copy.deepcopy(res.ori_sum_of_valence)
		valence_B = copy.deepcopy(res.ori_sum_of_valence)
		fake_super_atom_idx_list = list(set(res.ori_super_atom_idx_list).difference(set(res.exclude_super_atom_list)))
		#N = res.ori_n + plus_n - res.fake_n #this will lose the electric neutua
		N = abs(res.ori_n + res.plus_n)

		try:
			PATH = enumerate_path(N,fake_super_atom_idx_list,valence_A,res)
			lowest_energy_PATH_idx = get_the_operation_path_by_energy(PATH, fake_super_atom_idx_list, valence_A, res)
			while N != 0:
				N, valence_A = group_charge_transfer_by_path(PATH[lowest_energy_PATH_idx],N,fake_super_atom_idx_list,valence_A,res)
			res.ori_sum_of_valence = valence_A
		except:
			try:
				res.ori_sum_of_valence = put_fake_charge_back_to_super_atom(N, fake_super_atom_idx_list, valence_B ,res)

			except:
				res.ori_sum_of_valence = put_fake_charge_back_to_link_atom(N, res.link_atom_idx_list, valence_B ,res)

		for pv in res.perfect_valence_list:
			res.ori_sum_of_valence[pv[0]] = pv[1]
		res.sum_of_valence = res.ori_sum_of_valence
"""END HERE"""