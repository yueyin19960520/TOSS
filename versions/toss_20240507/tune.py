from post_process import *
import copy
import itertools
import math
import numpy as np
import random

class TUNE():
	def __init__(self):
		None

	def apply_resonance(self, atom_idx_list, valence_list, LOSS, likelyhood, prior, res, global_nomalized_dict, 
																	  					 global_sigma_dict, 
																	  					 global_mean_dict):
		ori_LOSS = copy.deepcopy(LOSS)
		ori_LIKELYHOOD = copy.deepcopy(likelyhood)
		ori_PRIOR = copy.deepcopy(prior)

		classes_dict = classify(atom_idx_list,res)
		valence_with_loss = []
		NC = len(classes_dict)
		for c in classes_dict:
			for d in classes_dict:
				if c[0] != d[0]:
					op_c = classes_dict[c]
					op_d = classes_dict[d]

					now_valence_list_c = [valence_list[i] for i in op_c]
					now_max_oxi_list_c = [res.max_oxi_list[i] for i in op_c]
					now_min_oxi_list_c = [max(0,res.min_oxi_list[i]) for i in op_c]   #because they are the super atom!
					now_valence_list_d = [valence_list[i] for i in op_d]
					now_max_oxi_list_d = [res.max_oxi_list[i] for i in op_d]
					now_min_oxi_list_d = [max(0,res.min_oxi_list[i]) for i in op_d]   #same reason.

					if len(op_c) < len(op_d):
						now_valence_list_c = [i + 1 for i in now_valence_list_c]
						N = len(op_c)
						while N != 0:
							temp_list = sorted([(j, v) for j,v in enumerate(now_valence_list_d)], reverse = True, key = lambda x:x[1])
							now_valence_list_d[temp_list[0][0]] -= 1
							N -= 1
							#print("38", N, res.mid)

						target_valence = set(now_valence_list_d)

						#make sure the volume of the memory!
						total_combi = math.factorial(len(op_d))/math.factorial(len(op_d)-len(op_c))/math.factorial(len(op_c))
						while total_combi >= 10000000:
							op_d = random.sample(op_d, round(len(op_d)*(4/5)))
							total_combi = math.factorial(len(op_d))/math.factorial(len(op_d)-len(op_c))/math.factorial(len(op_c))
							#print("47", total_combi, res.mid)

						possible_combi = []
						for p in itertools.combinations(op_d, len(op_c)):
							possible_combi.append(p)

						num_sample = round(3000/(NC**2)) if int(res.resonance_order) < 2 else round((3000)**(1/2)/(NC**2))
						if len(possible_combi) >= num_sample:
							possible_combi = random.sample(possible_combi, num_sample)

						resonance = {"op_d": possible_combi, "target_valence":target_valence}
						

					if len(op_c) > len(op_d):
						now_valence_list_d = [j - 1 for j in now_valence_list_d]
						N = len(op_d)
						while N != 0:
							temp_list = sorted([(i, v) for i,v in enumerate(now_valence_list_c)], reverse = False, key = lambda x:x[1])
							now_valence_list_c[temp_list[0][0]] += 1
							N -= 1
							#print("65", N, res.mid)
                        
						target_valence = set(now_valence_list_c)

						#make sure the volume of the memory!
						total_combi = math.factorial(len(op_c))/math.factorial(len(op_c)-len(op_d))/math.factorial(len(op_d))
						while total_combi >= 10000000:
							op_c = random.sample(op_c, round(len(op_c)*(4/5)))
							total_combi = math.factorial(len(op_c))/math.factorial(len(op_c)-len(op_d))/math.factorial(len(op_d))
							#print("74", total_combi, res.mid)
						possible_combi = []

						for p in itertools.combinations(op_c, len(op_d)):
							possible_combi.append(p)

						num_sample = round(3000/(NC**2)) if int(res.resonance_order) < 2 else round((3000)**(1/2)/(NC**2))
						if len(possible_combi) >= num_sample:
							possible_combi = random.sample(possible_combi, num_sample)

						resonance = {"op_c": possible_combi,"target_valence":target_valence}
						
				
					if len(op_c) == len(op_d):
						now_valence_list_c = [i + 1 for i in now_valence_list_c]
						now_valence_list_d = [j - 1 for j in now_valence_list_d]
						resonance = {"equal": [tuple(op_c)]}

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

						if list(resonance.keys())[0] == "equal":
							temp_sum_of_valence = copy.deepcopy(valence_list)
							for i in op_c:
								temp_sum_of_valence[i] += 1
							for j in op_d:
								temp_sum_of_valence[j] -= 1
							possible_valence_list = [temp_sum_of_valence] 

						for vl in possible_valence_list:
							temp_pair_info = spider_pair_length_with_CN_unnorm(vl, res)
							temp_LOSS, temp_LIKELYHOOD, temp_PRIOR = cal_loss_func_by_MAP(temp_pair_info,global_nomalized_dict, global_sigma_dict, global_mean_dict)
							valence_with_loss.append((vl,temp_LOSS,temp_LIKELYHOOD, temp_PRIOR))

					else:
						valence_with_loss.append((valence_list, ori_LOSS, ori_LIKELYHOOD, ori_PRIOR))

		return valence_with_loss


	def tune_by_resonance(self, LOSS, likelyhood, prior, res, global_nomalized_dict, global_sigma_dict, global_mean_dict):
		valence_with_loss = self.apply_resonance(res.super_atom_idx_list, 
												 res.sum_of_valence, 
												 LOSS, 
												 likelyhood,
												 prior,
												 res,
												 global_nomalized_dict, 
												 global_sigma_dict, 
												 global_mean_dict)  #first run
		#print(valence_with_loss)
		check = {}
		#print(valence_with_loss)
		for vwl in valence_with_loss:
			if str(sorted(vwl[0])) not in check:
				check[str(sorted(vwl[0]))] = [vwl]
			else:
				if vwl not in check[str(sorted(vwl[0]))]:
					check[str(sorted(vwl[0]))].append(vwl)

		if res.resonance_order == "2":
			for vwl in valence_with_loss:
				prime_valence_with_loss = self.apply_resonance(res.super_atom_idx_list, 
															   vwl[0], 
															   vwl[1], 
															   vwl[2],
															   vwl[3],
															   res, 
															   global_nomalized_dict, 
															   global_sigma_dict, 
															   global_mean_dict)
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


###########################################################################################################
###################################################OLS APPROACH############################################
###########################################################################################################
	def apply_redox_by_OLS(self, classes_dict, valence_list, res,global_nomalized_dict):
		loss_matrix_dict = {}
		for c in classes_dict: #LOSE electron, valence plus 1
			for d in classes_dict: #get electron, valence minus 1
				if c[0] != d[0]:
					now_valence_list_i = [valence_list[i] for i in classes_dict[c]]
					now_max_oxi_list_i = [res.max_oxi_list[i] for i in classes_dict[c]]
					now_min_oxi_list_i = [max(1,res.min_oxi_list[i]) for i in classes_dict[c]]  #because the center atom must > 0, right!?
					now_valence_list_j = [valence_list[i] for i in classes_dict[d]]
					now_max_oxi_list_j = [res.max_oxi_list[i] for i in classes_dict[d]]
					now_min_oxi_list_j = [max(1,res.min_oxi_list[i]) for i in classes_dict[d]]  #because the center atom must > 0, right!?

					num_c = len(classes_dict[c])
					num_d = len(classes_dict[d])
					lcm = num_c*num_d/math.gcd(num_c,num_d)
					num_lose = int(lcm/num_c)
					num_get = int(lcm/num_d)
					now_valence_list_i = [i + num_lose for i in now_valence_list_i]
					now_valence_list_j = [i - num_get for i in now_valence_list_j]

					if all([now_max_oxi_list_i[i] >= now_valence_list_i[i] for i in range(len(classes_dict[c]))]) and all([now_min_oxi_list_j[i] <= now_valence_list_j[i] for i in range(len(classes_dict[d]))]):
						for i in classes_dict[c]:
							valence_list[i] += num_lose
						for i in classes_dict[d]:
							valence_list[i] -= num_get

						try:
							temp_pair_info = spider_pair_length_with_CN_unnorm(valence_list, res)
							LOSS = cal_loss_func_by_OLS(temp_pair_info,global_nomalized_dict)

						except:
							LOSS = 100000
						loss_matrix_dict[LOSS] = copy.deepcopy(valence_list)
						for i in classes_dict[c]:
							valence_list[i] -= num_lose
						for i in classes_dict[d]:
							valence_list[i] += num_get
					else:
						temp_pair_info = spider_pair_length_with_CN_unnorm(valence_list, res)
						LOSS = cal_loss_func_by_OLS(temp_pair_info,global_nomalized_dict)
						loss_matrix_dict[LOSS] = valence_list
				else:
					temp_pair_info = spider_pair_length_with_CN_unnorm(valence_list, res)
					LOSS = cal_loss_func_by_OLS(temp_pair_info,global_nomalized_dict)
					loss_matrix_dict[LOSS] = valence_list
		return loss_matrix_dict


	def tune_by_redox_in_certain_range_by_OLS(self, atom_idx_list, LOSS, valence_list, bound, res, global_normalized_normed_dict):
		classes_dict = classify(atom_idx_list,res)
		ori_LOSS = copy.deepcopy(LOSS)
		ori_sum_of_valence = copy.deepcopy(valence_list)

		loss_matrix_dict = self.apply_redox_by_OLS(classes_dict, valence_list, res, global_normalized_normed_dict)

		LOSS = min(loss_matrix_dict.keys())
		valence_list = loss_matrix_dict[LOSS]
		if abs((LOSS - ori_LOSS) / ori_LOSS) > bound:
			return LOSS, valence_list
		else:
			return ori_LOSS, ori_sum_of_valence


###########################################################################################################
###################################################MLE APPROACH############################################
###########################################################################################################
	def apply_redox_by_MLE(self, classes_dict, valence_list, res, global_normalized_normed_dict):
		loss_matrix_dict = {}
		for c in classes_dict: #LOSE electron, valence plus 1
			for d in classes_dict: #get electron, valence minus 1
				if c[0] != d[0]:
					now_valence_list_i = [valence_list[i] for i in classes_dict[c]]
					now_max_oxi_list_i = [res.max_oxi_list[i] for i in classes_dict[c]]
					now_min_oxi_list_i = [max(1,res.min_oxi_list[i]) for i in classes_dict[c]]  #because the center atom must > 0, right!?
					now_valence_list_j = [valence_list[i] for i in classes_dict[d]]
					now_max_oxi_list_j = [res.max_oxi_list[i] for i in classes_dict[d]]
					now_min_oxi_list_j = [max(1,res.min_oxi_list[i]) for i in classes_dict[d]]  #because the center atom must > 0, right!?

					num_c = len(classes_dict[c])
					num_d = len(classes_dict[d])
					lcm = num_c*num_d/math.gcd(num_c,num_d)
					num_lose = int(lcm/num_c)
					num_get = int(lcm/num_d)
					now_valence_list_i = [i + num_lose for i in now_valence_list_i]
					now_valence_list_j = [i - num_get for i in now_valence_list_j]

					if all([now_max_oxi_list_i[i] >= now_valence_list_i[i] for i in range(len(classes_dict[c]))]) and all([now_min_oxi_list_j[i] <= now_valence_list_j[i] for i in range(len(classes_dict[d]))]):
						for i in classes_dict[c]:
							valence_list[i] += num_lose
						for i in classes_dict[d]:
							valence_list[i] -= num_get

						try:
							temp_pair_info = spider_pair_length_with_CN_unnorm(valence_list, res)
							LOSS = cal_loss_func_by_MLE(temp_pair_info,global_normalized_normed_dict)

						except:
							LOSS = 100000
						loss_matrix_dict[LOSS] = copy.deepcopy(valence_list)
						for i in classes_dict[c]:
							valence_list[i] -= num_lose
						for i in classes_dict[d]:
							valence_list[i] += num_get
					else:
						temp_pair_info = spider_pair_length_with_CN_unnorm(valence_list, res)
						LOSS = cal_loss_func_by_MLE(temp_pair_info, global_normalized_normed_dict)
						loss_matrix_dict[LOSS] = valence_list
				else:
					temp_pair_info = spider_pair_length_with_CN_unnorm(valence_list, res)
					#print(temp_pair_info)
					LOSS = cal_loss_func_by_MLE(temp_pair_info,global_normalized_normed_dict)
					loss_matrix_dict[LOSS] = valence_list
		#print(loss_matrix_dict)
		return loss_matrix_dict


	def tune_by_redox_in_certain_range_by_MLE(self, atom_idx_list, LOSS, valence_list, bound, res,global_normalized_normed_dict):
		classes_dict = classify(atom_idx_list,res)
		ori_LOSS = copy.deepcopy(LOSS)
		ori_sum_of_valence = copy.deepcopy(valence_list)

		loss_matrix_dict = self.apply_redox_by_MLE(classes_dict, valence_list, res, global_normalized_normed_dict)
		#print(loss_matrix_dict)
		LOSS = max(loss_matrix_dict.keys())
		valence_list = loss_matrix_dict[LOSS]
		if abs((LOSS - ori_LOSS) / ori_LOSS) > bound:
			return LOSS, valence_list
		else:
			return ori_LOSS, ori_sum_of_valence


###########################################################################################################
###################################################MAP APPROACH############################################
###########################################################################################################
	def apply_redox_by_MAP(self, classes_dict, valence_list, res, global_normalized_normed_dict,global_sigma_dict, global_mean_dict):
		loss_matrix_dict = {}
		for c in classes_dict: #LOSE electron, valence plus 1
			for d in classes_dict: #get electron, valence minus 1
				if c[0] != d[0]:
					now_valence_list_i = [valence_list[i] for i in classes_dict[c]]
					now_valence_list_j = [valence_list[i] for i in classes_dict[d]]
					if c[2] == "super":
						now_max_oxi_list_i = [res.max_oxi_list[i] for i in classes_dict[c]]
						now_min_oxi_list_i = [max(1,res.min_oxi_list[i]) for i in classes_dict[c]]  #because the center atom must > 0, right!?
					else: #c[2] == "link"
						now_min_oxi_list_i = [res.min_oxi_list[i] for i in classes_dict[c]]
						now_max_oxi_list_i = [min(0, res.max_oxi_list[i]) for i in classes_dict[c]]
					if d[2] == "super":
						now_max_oxi_list_j = [res.max_oxi_list[i] for i in classes_dict[d]]
						now_min_oxi_list_j = [max(1,res.min_oxi_list[i]) for i in classes_dict[d]]  #because the center atom must > 0, right!?
					else:#d[2] == "link"
						now_min_oxi_list_j = [res.min_oxi_list[i] for i in classes_dict[d]]
						now_max_oxi_list_j = [min(0, res.max_oxi_list[i]) for i in classes_dict[d]]						

					num_c = len(classes_dict[c])
					num_d = len(classes_dict[d])
					lcm = num_c*num_d/math.gcd(num_c,num_d)
					num_lose = int(lcm/num_c)
					num_get = int(lcm/num_d)
					now_valence_list_i = [i + num_lose for i in now_valence_list_i]
					now_valence_list_j = [i - num_get for i in now_valence_list_j]

					if all([now_max_oxi_list_i[i] >= now_valence_list_i[i] for i in range(len(classes_dict[c]))]) and all([now_min_oxi_list_j[i] <= now_valence_list_j[i] for i in range(len(classes_dict[d]))]):
						for i in classes_dict[c]:
							valence_list[i] += num_lose
						for i in classes_dict[d]:
							valence_list[i] -= num_get

						#try:
						if True:
							temp_pair_info = spider_pair_length_with_CN_unnorm(valence_list, res)
							LOSS,avg_likelyhood,avg_prior = cal_loss_func_by_MAP(temp_pair_info,global_normalized_normed_dict, global_sigma_dict, global_mean_dict)

						#except:
						else:
							LOSS = 100000
						loss_matrix_dict[LOSS] = (copy.deepcopy(valence_list), copy.deepcopy(avg_likelyhood), copy.deepcopy(avg_prior))
						for i in classes_dict[c]:
							valence_list[i] -= num_lose
						for i in classes_dict[d]:
							valence_list[i] += num_get
					else:

						temp_pair_info = spider_pair_length_with_CN_unnorm(valence_list, res)
						LOSS,avg_likelyhood,avg_prior = cal_loss_func_by_MAP(temp_pair_info, global_normalized_normed_dict, global_sigma_dict, global_mean_dict)
						loss_matrix_dict[LOSS] = (valence_list,avg_likelyhood,avg_prior)
				else:
					temp_pair_info = spider_pair_length_with_CN_unnorm(valence_list, res)
					#print(temp_pair_info)
					LOSS,avg_likelyhood,avg_prior = cal_loss_func_by_MAP(temp_pair_info,global_normalized_normed_dict, global_sigma_dict, global_mean_dict)
					loss_matrix_dict[LOSS] = (valence_list,avg_likelyhood,avg_prior)
		#print(loss_matrix_dict)
		return loss_matrix_dict


	def tune_by_redox_in_certain_range_by_MAP(self, atom_idx_list, LOSS, likelyhood, prior, valence_list, bound, res, global_normalized_normed_dict, global_sigma_dict, global_mean_dict):
		classes_dict = classify(atom_idx_list,res)
		ori_LOSS = copy.deepcopy(LOSS)
		ori_likelyhood = copy.deepcopy(likelyhood)
		ori_prior = copy.deepcopy(prior)
		ori_sum_of_valence = copy.deepcopy(valence_list)

		loss_matrix_dict = self.apply_redox_by_MAP(classes_dict, 
												   valence_list, 
												   res, 
												   global_normalized_normed_dict, 
												   global_sigma_dict, 
												   global_mean_dict)

		LOSS = min(loss_matrix_dict.keys())
		valence_list = loss_matrix_dict[LOSS][0]
		likelyhood = loss_matrix_dict[LOSS][1]
		prior = loss_matrix_dict[LOSS][2]

		if abs((LOSS - ori_LOSS) / ori_LOSS) > bound:
			return LOSS, likelyhood, prior, valence_list
		else:
			return ori_LOSS, ori_likelyhood, ori_prior, ori_sum_of_valence
"""END HERE"""
