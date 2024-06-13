from second_algo import enu_permu_combi, enumerate_path, group_charge_transfer_by_path, get_the_operation_path_by_energy 
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


	def apply_resonance(self, atom_idx_list, LOSS, valence_list, res):
		ori_LOSS = copy.deepcopy(LOSS)
		classes_dict = classify(atom_idx_list)
		valence_with_loss = []
		for c in classes_dict:
			for d in classes_dict:
				if c[0] != d[0]:
					op_c = classes_dict[c]
					op_d = classes_dict[d]

					now_valence_list_c = [valence_list[i] for i in op_c]
					now_max_oxi_list_c = [res.max_oxi_list[i] for i in op_c]
					now_min_oxi_list_c = [max(1,res.min_oxi_list[i]) for i in op_c]   #because they are the super atom!
					now_valence_list_d = [valence_list[i] for i in op_d]
					now_max_oxi_list_d = [res.max_oxi_list[i] for i in op_d]
					now_min_oxi_list_d = [max(1,res.min_oxi_list[i]) for i in op_d]   #same reason.

					if len(op_c) < len(op_d):
						now_valence_list_c = [i + 1 for i in now_valence_list_c]
						N = len(op_c)
						while N != 0:
							temp_list = sorted([(j, v) for j,v in enumerate(now_valence_list_d)], reverse = True, key = lambda x:x[1])
							now_valence_list_d[temp_list[0][0]] -= 1
							N -= 1
							print("125")

						target_valence = set(now_valence_list_d)
						possible_combi = []
						for p in itertools.combinations(op_d, len(op_c)):
							possible_combi.append(p)

						resonance = {"op_d": possible_combi, "target_valence":target_valence}

					if len(op_c) > len(op_d):
						now_valence_list_d = [j - 1 for j in now_valence_list_d]
						N = len(op_d)
						while N != 0:
							temp_list = sorted([(i, v) for i,v in enumerate(now_valence_list_c)], reverse = False, key = lambda x:x[1])
							now_valence_list_c[temp_list[0][0]] += 1
							N -= 1
							print("141")
                        
						target_valence = set(now_valence_list_c)
						possible_combi = []
						for p in itertools.combinations(op_c, len(op_d)):
							possible_combi.append(p)

						resonance = {"op_c": possible_combi,"target_valence":target_valence}
                
					if len(op_c) == len(op_d):
						now_valence_list_c = [i + 1 for i in now_valence_list_c]
						now_valence_list_d = [j - 1 for j in now_valence_list_d]

					if all([now_max_oxi_list_c[i] >= now_valence_list_c[i] for i in range(len(op_c))]) and all(
							[now_min_oxi_list_d[i] <= now_valence_list_d[i] for i in range(len(op_d))]):
	
						resonance_result = []
						if list(resonance.keys())[0] == "op_d":
							possible_valence_list = []
							for p in resonance["op_d"]:
								temp_sum_of_valence = copy.deepcopy(valence_list)
								for i in op_c:
									temp_sum_of_valence[i] += 1
								for j in p:
									temp_sum_of_valence[j] -= 1
								if set([temp_sum_of_valence[i] for i in op_d]) == resonance["target_valence"]:
									possible_valence_list.append(temp_sum_of_valence)                                         

						if list(resonance.keys())[0] == "op_c":
							possible_valence_list = []
							for p in resonance["op_c"]:
								temp_sum_of_valence = copy.deepcopy(valence_list)
								for i in p:
									temp_sum_of_valence[i] += 1
								for j in op_d:
									temp_sum_of_valence[j] -= 1
								if set([temp_sum_of_valence[i] for i in op_c]) == resonance["target_valence"]:
									possible_valence_list.append(temp_sum_of_valence)
	
						for vl in possible_valence_list:
							temp_pair_info = spider_pair_length_with_CN_unnorm(res)
							temp_LOSS = cal_loss_func_with_CN(temp_pair_info,res)
							valence_with_loss.append((vl,temp_LOSS))

					else:
						valence_with_loss.append((valence_list,ori_LOSS))

		return valence_with_loss


	def tune_by_resonance(self, LOSS, res):
		valence_with_loss = apply_resonance(res.super_atom_idx_list, res.sum_of_valence, LOSS)  #first run
		check = {}

		for vwl in valence_with_loss:
			if str(sorted(vwl[0])) not in check:
				check[str(sorted(vwl[0]))] = [vwl]
			else:
				if vwl not in check[str(sorted(vwl[0]))]:
					check[str(sorted(vwl[0]))].append(vwl)

		if res.resonance_order == "2":
			for vwl in valence_with_loss:
				prime_valence_with_loss = apply_resonance(res.super_atom_idx_list, vwl[0], vwl[1])
				for p in prime_valence_with_loss:
					if str(sorted(p[0])) not in check:
						check[str(sorted(p[0]))] = [p]
					else:
						if p not in check[str(sorted(p[0]))]:
							check[str(sorted(p[0]))].append(p)

		possible_resonance = {}
		for k,v in check.items():
			avg_loss = sum(l[1] for l in v)/len(v)
			possible_resonance[avg_loss] = v

		key = sorted(possible_resonance.keys())[0]

		the_resonance_result = possible_resonance[key]
		avg_LOSS = sum([i[1] for i in the_resonance_result])/len(the_resonance_result)
		#one_sum_of_valence = the_resonance_result[0][0]
		resonance_valence_list = [i[0] for i in the_resonance_result]
		return avg_LOSS, the_resonance_result
"""END HERE"""