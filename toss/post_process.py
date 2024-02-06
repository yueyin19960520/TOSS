from digest import get_ele_from_sites
from result import RESULT
from multiprocessing.dummy import Pool as ThreadPool
import copy
import math
import numpy as np

def tolerance_corr(func, m_id, delta_X, tolerance_list):
	tolerance_trial = tolerance_list
	single_result_dict_normed = {}
	single_result_dict = {}
	single_super_point_dict = {}
	for t in tolerance_trial:
		try:
		#if True:
			res = RESULT()
			res.mid = m_id
			func(m_id, delta_X, t, tolerance_list, res)
			temp_pair_info_normed = spider_pair_length_with_CN_normed(res)
			temp_pair_info = spider_bond_length(res)
			single_result_dict[t] = temp_pair_info
			single_result_dict_normed[t] = temp_pair_info_normed
			super_point_list = [[get_ele_from_sites(i,res), sorted(res.shell_ele_list[i])] for i in res.idx]
			single_super_point_dict[t] = super_point_list
		except:
		#else:
			LOSS = None
			temp_pair_info = None
			temp_pair_info_normed = None
			single_super_point_dict = None
	return single_result_dict_normed, single_result_dict, single_super_point_dict #They could be empty.


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


def cut_the_work_list_prime(meet, N):
    Npiece = N
    piece_length = len(meet)//Npiece
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


def cal_loss_func_by_MAP(temp_pair_info,pred_dict, global_sigma_dict, global_mean_dict):
	n = sum([len(vv) for v in temp_pair_info.values() for vv in v.values()])

	sum_likelyhood = 0
	prior = 0

	for pair_name,info in temp_pair_info.items():
		#NL = sum([i[0] for i in useful_pair.values()])
		if pair_name in pred_dict:
			useful_pair = pred_dict[pair_name]
			for label,length_list in info.items():

				#it is the likelyhood calculation:
				NL = sum([v[1] for k,v in useful_pair.items() if k[0] == label[0]])
				if NL == 0:
					NL = sum([v[1] for k,v in useful_pair.items()])
				try:
					nl = useful_pair[label][1]
				except:
					nl = 1
				likelyhood = len(length_list) * math.log(nl/NL)
				sum_likelyhood += likelyhood
				#print("likelyhood:%s"%likelyhood)

				#it is the prior probability calculation:
				key = (pair_name, label[0], label[1])
				try:
					mean = round(global_mean_dict[key],3)
					sigma = round(global_sigma_dict[key],3)
					sigma = 0.01 if sigma == 0 else sigma
				except:
					possible_keys = [k for k in global_mean_dict.keys() if k[0] == pair_name]
					mean = np.mean([global_mean_dict[key] for key in possible_keys])
					sigma = np.mean([global_sigma_dict[key] for key in possible_keys])

				sub_prior = 0
				for l in length_list:
					gx = (1/(np.sqrt(2*np.pi)*sigma)) * np.exp(-(round(l,3)-mean)**2/(2*sigma**2))
					gx_den = gx * 0.001
					math_domin_limit = 10**(-323.60)
					gx_den = gx_den if gx_den > math_domin_limit else math_domin_limit
					sub_prior += math.log(gx_den)
				prior += sub_prior
				#print(key)
				#print("prior:%s"%sub_prior)
		else:
			for label,length_list in info.items():
				NL = 100000
				nl = 1
				likelyhood = len(length_list) * math.log(nl/NL)
				sum_likelyhood += likelyhood
				raise ValueError("WRONG!")

	avg_likelyhood = -1*(sum_likelyhood/n)
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
"""END HERE"""


	
"""
def leave_one_out(m_id):
	matched_dict = {}
	#try:
	if True:
		useful_pairs = list(pairs_info_dict[m_id].keys())

		for mid,info in pairs_info_dict.items():
			if mid != m_id:
				for pair_name,label_info in info.items():
					if pair_name in useful_pairs:
						if pair_name not in matched_dict:
							matched_dict[pair_name] = {}
							for label,length_info in label_info.items():
								if label not in matched_dict[pair_name]:
									matched_dict[pair_name][label] = []
									matched_dict[pair_name][label].append(length_info)
								else:
									matched_dict[pair_name][label].append(length_info)
						else:
							for label,length_info in label_info.items():
								if label not in matched_dict[pair_name]:
									matched_dict[pair_name][label] = []
									matched_dict[pair_name][label].append(length_info)
								else:
									matched_dict[pair_name][label].append(length_info)
	                        
		for pair_name, info in matched_dict.items():
			for label, length_info in info.items():
				average_length = sum([(i[0]*i[1]) for i in length_info])/sum(i[1] for i in length_info)
				nomalized_num = sum(i[1] for i in length_info)
				matched_dict[pair_name][label] = (average_length, nomalized_num)  

		parameters = [m_id, matched_dict]  
		print(m_id, len(matched_dict))      
		return parameters
	#except:
		#parameters = [m_id, "FAILED"]
		#return parameters


def get_the_best_t(func, m_id, delta_X, tolerance_list, global_nomalized_dict):
	single_result_dict = {}
	for t in tolerance_list:
		try:
			res = RESULT()
			func(m_id, delta_X, t, res)
			N_spec = len(res.species_uni_list)
			temp_pair_info = spider_pair_length_with_CN_unnorm(res.sum_of_valence, res)
			LOSS = cal_loss_func_with_CN(temp_pair_info, global_nomalized_dict)
			single_result_dict[t] = (N_spec, LOSS)

		except:
			single_result_dict[t] = None
	
	tar = sorted([(k,v[1],(5**v[0])*v[1]) for k,v in single_result_dict.items() if v != None], key=lambda x:x[2])
	t,LOSS = tar[0][0],tar[0][1]
	return t,LOSS


def cal_loss_by_atom(res, vl, global_nomalized_dict):
	LOSS_list = []
	valence_list = copy.deepcopy(vl)
	for i in res.idx:
		super_atom_pair_info = {}
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

			#CN_name = SCN
			CN_name = pair_CN
			OS_name = pair_OS
			label = (CN_name, OS_name)			

			if pair_name not in super_atom_pair_info:
				super_atom_pair_info[pair_name] = {}
				if label not in super_atom_pair_info[pair_name]:
					super_atom_pair_info[pair_name][label] = [length_list[j]]
				else:
					super_atom_pair_info[pair_name][label].append(length_list[j])
			else:
				if label not in super_atom_pair_info[pair_name]:
					super_atom_pair_info[pair_name][label] = [length_list[j]]
				else:
					super_atom_pair_info[pair_name][label].append(length_list[j])

		loss = 0
		N = 0

		for pair_name,info in super_atom_pair_info.items():
			if pair_name in global_nomalized_dict:
				for label, length_list in info.items():
					if label in global_nomalized_dict[pair_name]:
						pred_length = global_nomalized_dict[pair_name][label][0]
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

		LOSS_per_atom = loss/N * 10000
		LOSS_list.append(LOSS_per_atom)
	return LOSS_list


def cal_loss_func_by_MLE(temp_pair_info, pred_dict):
	n = sum([len(vv) for v in temp_pair_info.values() for vv in v.values()])

	sum_likelyhood = 0
	for pair_name,info in temp_pair_info.items():
		useful_pair = pred_dict[pair_name]
		#NL = sum([i[0] for i in useful_pair.values()])
		if pair_name in pred_dict:
			for label,length_list in info.items():
				NL = sum([v[1] for k,v in useful_pair.items() if k[0] == label[0]])
				try:
					nl = useful_pair[label][1]
				except:
					nl = 1
				likelyhood = len(length_list) * math.log(nl/NL)
				sum_likelyhood += likelyhood
		else:
			for label,length_list in info.items():
				NL = 100000
				nl = 1
				likelyhood = len(length_list) * math.log(nl/NL)
				sum_likelyhood += likelyhood

	avg_likelyhood = sum_likelyhood/n
	return avg_likelyhood
"""

"""
def leave_one_out(target_group, pairs_info_dict,global_nomalized_dict):
    global_matched_dict = {}
    for m_id in target_group:
        try:
            target = pairs_info_dict[m_id]
            #matched_dict = copy.deepcopy({k:{l:list(global_nomalized_dict[k][(l[0], l[1])]) for l in global_nomalized_dict[k].keys() if l[0] in [i[0] for i in target[k].keys()]} for k in target.keys()})
            #matched_dict = copy.deepcopy({k:global_nomalized_dict[k]} for k in target.keys())

            for pair_name, info in target.items():
                for label, sub_info in info.items():
                    N = matched_dict[pair_name][label][1]
                    n = sub_info[1]
                    L = matched_dict[pair_name][label][0]
                    l = sub_info[0]
                    matched_dict[pair_name][label][1] = round((N-n),2)
                    if matched_dict[pair_name][label][1] != 0:
                        matched_dict[pair_name][label][0] = ((N*L)-(n*l))/(N-n)
                    else:
                        del(matched_dict[pair_name][label])
            global_matched_dict.update({m_id:matched_dict})
        except:
            None
    return global_matched_dictv

global_matched_dict = leave_one_out(target_group, pairs_info_dict)
file_save = open("D:/share/ML for CoreLevel/GFOS/%s/global_matched_dict_%s.pkl"%(database,date),'wb') 
pickle.dump(global_matched_dict, file_save) 
file_save.close()


def leave_one_out(target_group, pairs_info_dict):
    global_matched_dict = {}
    for mid in target_group:
        if mid in pairs_info_dict:
            useful_pair_name = set([l for t in pairs_info_dict[mid].keys() for l in pairs_info_dict[mid][t].keys()])
            length_info_dict = {k:{} for k in useful_pair_name}
            for MID, single_info in pairs_info_dict.items():
                if MID != mid:
                    Nt = (len(single_info))
                    for t_value, real_info in single_info.items():
                        for pair_name, info in real_info.items():
                            if pair_name in useful_pair_name:
                                for label,length_list in info.items():
                                    if label in length_info_dict[pair_name]:
                                        length_info_dict[pair_name][label].append([1/Nt,length_list])
                                    else:
                                        length_info_dict[pair_name][label] = [[1/Nt,length_list]]

            for pair_name, info in length_info_dict.items():
                for label, length_list in info.items():

                    sum_of_length = sum([l*v[0] for v in length_list for l in v[1]])

                    sum_of_number = sum([v[0]*len(v[1]) for v in length_list])

                    mean = sum_of_length/sum_of_number

                    temp = []
                    for v in length_list:
                        for l in v[1]:
                            temp.append(v[0] * (l - mean)**2)
                    sum_of_length_square = sum(temp)
                    sigma = np.sqrt(sum_of_length_square/sum_of_number)

                    length_info_dict[pair_name][label] = (mean, sum_of_number, sigma)
            global_matched_dict.update({mid:length_info_dict})
        else:
            None
    return global_matched_dict
"""