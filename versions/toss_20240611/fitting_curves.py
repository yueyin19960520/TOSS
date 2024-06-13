import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec
import pandas as pd
import re
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial 
import pickle
import time
import math
import sys
from result import RESULT
from pre_set import PRE_SET
from digest import DIGEST
from get_structure import GET_STRUCTURE
from digest import get_ele_from_sites
from post_process import *



def classify_length(res):
    pair_dict = {}
    for i in res.idx:
        for j in res.shell_idx_list[i]:
            ele_i = get_ele_from_sites(i,res)
            ele_j = get_ele_from_sites(j,res)

            if res.periodic_table.elements_list.index(ele_i) < res.periodic_table.elements_list.index(ele_j):
                pair_name = (ele_i, ele_j)
            else:
                pair_name = (ele_j, ele_i)

            if pair_name not in pair_dict:
                pair_dict[pair_name] = [res.matrix_of_length[i][j]]
            else:
                pair_dict[pair_name].append(res.matrix_of_length[i][j])
    return pair_dict


def save_bond_length(pair_name, length_list, save_path):
    with open(save_path + "bond_of_%s_%s.csv"%(pair_name[0],pair_name[1]), "a+") as f:
        for l in length_list:
            f.write(str(l)+"\n")
        f.close()


def global_classify(pair_info_dict):
    global_pairs_info = {}
    for k, pair_info in pair_info_dict.items():
        for pair_name, length_list in pair_info.items():
            if pair_name not in global_pairs_info:
                global_pairs_info[pair_name] = length_list
            else:
                global_pairs_info[pair_name] += length_list
    return global_pairs_info


def Spider(m_id,i,t):
    res = RESULT()
    PS = PRE_SET(spider = True)
    res.dict_ele, res.matrix_of_threshold = PS.dict_ele, PS.matrix_of_threshold
    res.Forced_transfer_group,res.inorganic_group = PS.Forced_transfer_group,PS.inorganic_group
    #res.transit_metals, res.metals = PS.transit_matals, PS.metals

    GS = GET_STRUCTURE(m_id)
    res.sites,res.idx, res.struct = GS.sites, GS.idx, GS.struct
    res.matrix_of_length, res.valence_list, res.elements_list = GS.matrix_of_length, GS.valence_list, GS.elements_list

    valid_t = [1.1] if t == 1.1 else [1.1, t]
    DG = DIGEST(valid_t, t, m_id, res)
    res.max_oxi_list, res.min_oxi_list = DG.max_oxi_list, DG.min_oxi_list
    res.SHELL_idx_list, res.threshold_list = DG.SHELL_idx_list, DG.threshold_list
    res.organic_patch, res.alloy_flag = DG.organic_patch, DG.alloy_flag
    DG.digest_structure_with_image(res)
    res.shell_ele_list, res.shell_env_list, res.shell_idx_list, res.shell_CN_list, res.shell_X_list = DG.shell_ele_list, DG.shell_env_list, DG.shell_idx_list, DG.shell_CN_list, DG.shell_X_list

    pair_dict = classify_length(res)
    parameter = [m_id, pair_dict]
    print('This is the %sth structure with mid %s and we got %s different pairs.'%(i, m_id, len(pair_dict)))
    return parameter


def Global_Spider(m_id,i,t):
    res = RESULT()
    PS = PRE_SET(spider = True, work_type = "global")
    res.dict_ele, res.matrix_of_threshold = PS.dict_ele, PS.matrix_of_threshold
    res.Forced_transfer_group,res.inorganic_group = PS.Forced_transfer_group,PS.inorganic_group
    #res.transit_metals, res.metals = PS.transit_matals, PS.metals

    GS = GET_STRUCTURE(m_id)
    res.sites,res.idx, res.struct = GS.sites, GS.idx, GS.struct
    res.matrix_of_length, res.valence_list, res.elements_list = GS.matrix_of_length, GS.valence_list, GS.elements_list
    
    valid_t = [1.1] if t == 1.1 else [1.1, t]
    DG = DIGEST(valid_t, t, m_id, res)
    res.max_oxi_list, res.min_oxi_list = DG.max_oxi_list, DG.min_oxi_list
    res.SHELL_idx_list, res.threshold_list = DG.SHELL_idx_list, DG.threshold_list
    res.organic_patch, res.alloy_flag = DG.organic_patch, DG.alloy_flag
    DG.digest_structure_with_image(res)
    res.shell_ele_list, res.shell_env_list, res.shell_idx_list, res.shell_CN_list, res.shell_X_list = DG.shell_ele_list, DG.shell_env_list, DG.shell_idx_list, DG.shell_CN_list, DG.shell_X_list

    pair_dict = classify_length(res)
    parameter = [m_id, pair_dict]
    print('This is the %sth structure with mid %s and we got %s different pairs.'%(i, m_id, len(pair_dict)))
    return parameter



def gaussian(x, amplified1, mean1, stddev1, amplified2, mean2, stddev2):
    return amplified1 * np.exp((-(x-mean1)**2)/(2*(stddev1**2))) + amplified2 * np.exp((-(x-mean2)**2)/(2*(stddev2**2)))


def test_r2(x, y, mean1, mean2, step, MIN,MAX):
    bounds = ([0,mean1-0.1*step, 0, 0,mean2-0.1*step, 0], 
              [0.2,mean1+0.1*step,(MAX-MIN)/8,1 ,mean2+0.1*step,(MAX-MIN)/8])
    popt, _ = curve_fit(gaussian, x, y, bounds = bounds)
    y_pred = [gaussian(X,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5]) for X in x]
    y_true = y
    R2 = r2_score(y_true,y_pred)
    RMSQ = mean_squared_error(y_true,y_pred, squared=False)
    return R2,popt,RMSQ,bounds


def s_gaussian(x, amplified1, mean1, stddev1):
    return amplified1 * np.exp((-(x-mean1)**2)/(2*(stddev1**2)))

def s_test_r2(x, y, mean1,step):
    bounds = ([0,mean1-0.1*step, 0], [0.2,mean1+0.1*step, 0.5])
    popt, _ = curve_fit(s_gaussian, x, y, bounds = bounds)
    y_pred = [s_gaussian(X,popt[0],popt[1],popt[2]) for X in x]
    y_true = y
    R2 = r2_score(y_true,y_pred)
    RMSQ = mean_squared_error(y_true,y_pred, squared=False)
    return R2,popt,RMSQ,bounds


def refine_data(Min, Max, step, target_list):
    """
    Adjusts values in target_list to the nearest lower station point defined by the range [Min, Max] and step.

    Parameters:
    - Min (float): Minimum value of the range.
    - Max (float): Maximum value of the range.
    - step (float): Step size between each station.
    - target_list (list): List of values to be refined.

    Returns:
    - list: List of refined values snapped to the nearest lower station.
    """
    # Create a list of stations rounded to three decimal places
    stations = np.arange(Min, Max + step, step)
    stations = np.round(stations, 3)

    # Function to find the nearest lower station for each value in target_list
    def find_station(value):
        # Find the largest station value that is less than or equal to the value
        station_indices = np.searchsorted(stations, value, side='right') - 1
        return stations[max(station_indices, 0)]  # Ensure index is within the bounds

    # Adjust each value in the target_list to the nearest station
    processed_list = [find_station(value) for value in target_list]

    return processed_list


def get_bond_length_distribution_fitting_info(ele1, ele2, former_threshold, refined_data, global_save_path):
    #plot the global bong length and its distribuion.
    path_for_all = global_save_path
    file_path = f"{path_for_all}bond_of_{ele1}_{ele2}.csv"
    alternate_file_path = f"{path_for_all}bond_of_{ele2}_{ele1}.csv"

    try:
        # Try to load the CSV file using the first element ordering
        global_data = pd.read_csv(file_path, header=None)
    except FileNotFoundError:
        print(f"File not found: bond_of_{ele1}_{ele2}.csv. Trying alternate file...")
        try:
            # Try to load the CSV file using the second element ordering
            global_data = pd.read_csv(alternate_file_path, header=None)
        except FileNotFoundError:
            global_data = []
            refined_data = []
            print(f"File not found: bond_of_{ele2}_{ele1}.csv. No data available for element pair {ele1}-{ele2} or {ele2}-{ele1}.")
        except Exception as e:
            print(f"An unexpected error occurred while trying to read bond_of_{ele2}_{ele1}.csv: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while trying to read bond_of_{ele1}_{ele2}.csv: {e}")

    if len(refined_data) != 0:
        list_of_distance = global_data[0].tolist() #convert data to list
        sorted_list_of_distance = sorted(list_of_distance)
        x_scatter = [i for i in range(len(list_of_distance))]

        g_2decimal_distance = [round(i, 2) for i in list_of_distance if 0.1 < i]
        g_result = pd.value_counts(g_2decimal_distance)
        g_temp_x = g_result.index.tolist()
        g_temp_y = g_result.values.tolist()
        total_samples = sum(g_temp_y)
        g_temp_xy = [[x,y/total_samples] for x,y in zip(g_temp_x, g_temp_y)]
        g_sorted_xy = sorted(g_temp_xy, key = lambda x:x[0])
        g_x = [g_sorted_xy[l][0] for l in range(len(g_sorted_xy))]
        g_y = [g_sorted_xy[l][1] for l in range(len(g_sorted_xy))]

        list_of_distance = [l for l in refined_data if former_threshold > l > 0.1]
        size = len(list_of_distance)
        sorted_list_of_distance = sorted(list_of_distance)
        x_scatter = [i for i in range(len(list_of_distance))]
        
        #MEAN = sum(list_of_distance)/len(list_of_distance)
        #refined_distance = [round(i, 2) for i in list_of_distance]
        Min, Max = min(list_of_distance), max(list_of_distance)
        Step = round(((Max-Min)/50),3)
        refined_distance = refine_data(Min, Max, Step, list_of_distance)
        
        result = pd.value_counts(refined_distance)
        temp_x = result.index.tolist()
        temp_y = result.values.tolist()
        total_refined_samples = sum(temp_y)
        temp_xy = [[x,y/total_refined_samples] for x,y in zip(temp_x, temp_y)]
        sorted_xy = sorted(temp_xy, key = lambda x:x[0])
        x = [sorted_xy[l][0] for l in range(len(temp_xy))]
        y = [sorted_xy[l][1] for l in range(len(temp_xy))]

        #Process the R2 and the RMSQ
        step = Step
        MIN = round(min(x),3) #min(mean1 - 3 * popt[2], mean2 - 3 * popt[5])
        MAX = round(max(x),3) #max(mean1 + 3 * popt[2], mean2 + 3 * popt[5])
        
        np_mean1, np_mean2 = np.arange(MIN, MAX, step),np.arange(MIN, MAX, step)
        dim = len(np_mean1)
        np_R2, np_RMSQ = np.zeros([dim,dim]),np.zeros([dim,dim])
        check_R2_list,check_mean_list = [],[]
        for i, mean1 in enumerate(np_mean1):
            for j, mean2 in enumerate(np_mean2):
                if i < j:
                    try:
                        R2,popt,RMSQ,bounds = test_r2(x, y, mean1, mean2, step, MIN,MAX)
                        R2 = 0 if R2 < 0 else R2
                        np_R2[i][j] = R2
                        np_R2[j][i] = R2
                        np_RMSQ[i][j] = RMSQ
                        np_RMSQ[j][i] = RMSQ
                        check_R2_list.append(R2)
                        check_mean_list.append([mean1,mean2])
                    except Exception as e:
                        print(f"An error occurred: {e}")
                        np_R2[i][j] = 0
                        np_R2[j][i] = 0
                        np_RMSQ[i][j] = max(np_RMSQ.flat)
                        np_RMSQ[j][i] = max(np_RMSQ.flat)
                if i == j:
                    try:
                        R2,popt,RMSQ,bounds = test_r2(x, y, mean1, mean2, step, MIN,MAX)
                        R2 = 0 if R2 < 0 else R2
                        np_R2[i][j] = R2
                        np_RMSQ[i][j] = RMSQ
                        check_R2_list.append(R2)
                        check_mean_list.append([mean1,mean2])
                    except Exception as e:
                        print(f"An error occurred: {e}")
                        np_R2[i][j] = 0
                        np_RMSQ[i][j] = max(np_RMSQ.flat)  
        best_mean_combo = [j for i,j in zip(check_R2_list, check_mean_list) if i == max(check_R2_list)][0]
        mean1, mean2 = best_mean_combo[0], best_mean_combo[1]
        R2,popt,RMSQ,bounds = test_r2(x, y, mean1, mean2, step, MIN,MAX)
        
        #the seventh plot at the bottom right, the fitting plots.
        x_perf = np.arange(MIN,MAX, 0.001)
        y_perf = [gaussian(X, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]) for X in x_perf]
        x_orig = [sorted_xy[l][0] for l in range(len(temp_xy)) if MIN<sorted_xy[l][0]<MAX]
        y_orig = [sorted_xy[l][1] for l in range(len(temp_xy)) if MIN<sorted_xy[l][0]<MAX]

        threshold = round(max(popt[1] + 4 * popt[2], popt[4] + 4 * popt[5]),3)
        sigma_1, sigma_2 = round(popt[2],3), round(popt[5],3)
        mean1, mean2 = round(popt[1],3), round(popt[4],3)
        result = {"R2":round(R2,2), 
                  "mean_1":mean1, "sigma_1": sigma_1, 
                  "mean_2":mean2, "sigma_2":sigma_2,
                  "size":size, 
                  "threshold":threshold}
        parameters = [(ele1, ele2), result]
    else:
        parameters = [(ele1, ele2), "None"]
    return parameters


def bond_length_distribution_fitting(ele1, ele2, former_threshold, refined_data, save_path, global_save_path):
    #plot the global bong length and its distribuion.
    path_for_all = global_save_path
    file_path = f"{path_for_all}bond_of_{ele1}_{ele2}.csv"
    alternate_file_path = f"{path_for_all}bond_of_{ele2}_{ele1}.csv"

    try:
        # Try to load the CSV file using the first element ordering
        global_data = pd.read_csv(file_path, header=None)
    except FileNotFoundError:
        print(f"File not found: bond_of_{ele1}_{ele2}.csv. Trying alternate file...")
        try:
            # Try to load the CSV file using the second element ordering
            global_data = pd.read_csv(alternate_file_path, header=None)
        except FileNotFoundError:
            global_data = []
            refined_data = []
            print(f"File not found: bond_of_{ele2}_{ele1}.csv. No data available for element pair {ele1}-{ele2} or {ele2}-{ele1}.")
        except Exception as e:
            print(f"An unexpected error occurred while trying to read bond_of_{ele2}_{ele1}.csv: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while trying to read bond_of_{ele1}_{ele2}.csv: {e}")
            
    if len(refined_data) != 0:
        fig = plt.figure(figsize=(20,22))
        gs = GridSpec(4,4)
        
        list_of_distance = global_data[0].tolist() #convert data to list
        #the first plot at top left of the top left, the global bondlength.
        ax1 = plt.subplot(gs[:1,:1])
        sorted_list_of_distance = sorted(list_of_distance)
        x_scatter = [i for i in range(len(list_of_distance))]
        ax1.plot(sorted_list_of_distance, x_scatter)
        ax1.set_title("The Distances between %s and %s."%(ele1,ele2), size = 12)
        plt.ylabel("Enumeration (meaningless)", size = 8, labelpad = 1)
        plt.xlabel("Distance (Å)", size = 8, labelpad = 1)

        #the second plot at top right of the top left, the global length distribution.
        g_2decimal_distance = [round(i, 2) for i in list_of_distance if 0.1 < i]
        g_result = pd.value_counts(g_2decimal_distance)
        g_temp_x = g_result.index.tolist()
        g_temp_y = g_result.values.tolist()
        total_samples = sum(g_temp_y)
        g_temp_xy = [[x,y/total_samples] for x,y in zip(g_temp_x, g_temp_y)]
        g_sorted_xy = sorted(g_temp_xy, key = lambda x:x[0])
        g_x = [g_sorted_xy[l][0] for l in range(len(g_sorted_xy))]
        g_y = [g_sorted_xy[l][1] for l in range(len(g_sorted_xy))]

        ax2 = plt.subplot(gs[0:1, 1:2])
        ax2.plot(g_x, g_y)
        ax2.set_title("The Distribution of %s and %s Distances."%(ele1,ele2), size = 12)
        plt.xlabel("Length (Å)", size = 8, labelpad = 1)
        plt.ylabel("Proportion (%)", size = 8, labelpad = 1)
        #finish processing the global data
        ###########################################################################################################
        
        list_of_distance = [l for l in refined_data if former_threshold >= l > 0.1] #3.079,4.36

        #the third plot at the left bottom of top left, the refined bond length info. 
        ax3 = plt.subplot(gs[1:2,0:1])
        sorted_list_of_distance = sorted(list_of_distance)
        x_scatter = [i for i in range(len(list_of_distance))]
        ax3.plot(sorted_list_of_distance,x_scatter)
        ax3.set_title("The Distances between %s and %s within shell."%(ele1,ele2), size = 12)
        plt.ylabel("Enumeration (meaningless)", size = 8, labelpad = 1)
        plt.xlabel("Distance (Å)", size = 8, labelpad = 1)
        
        #MEAN = sum(list_of_distance)/len(list_of_distance)
        #refined_distance = [round(i, 3) for i in list_of_distance] ##we can add a upper bound of the MEAN+2
        Min, Max = min(list_of_distance), max(list_of_distance)
        Step = round(((Max-Min)/50),3)
        refined_distance = refine_data(Min, Max, Step, list_of_distance)

        result = pd.value_counts(refined_distance)
        temp_x = result.index.tolist()
        temp_y = result.values.tolist()
        total_refined_samples = sum(temp_y)
        temp_xy = [[x,y/total_refined_samples] for x,y in zip(temp_x, temp_y)]
        sorted_xy = sorted(temp_xy, key = lambda x:x[0])
        x = [sorted_xy[l][0] for l in range(len(temp_xy))]
        y = [sorted_xy[l][1] for l in range(len(temp_xy))]

        #the fourth plot at the right bottom of the top left, the length distribution in the first coodinaiton shell.
        ax4 = plt.subplot(gs[1:2,1:2])
        ax4.plot(x,y)
        ax4.set_title("The Distribution of %s and %s Distances within shell."%(ele1,ele2), size = 12)
        plt.xlabel("Length (Å)", size = 8, labelpad = 1)
        plt.ylabel("Proportion (%)", size = 8, labelpad = 1)

        #Process the R2 and the RMSQ
        step = Step
        MIN = round(min(x),3) #min(mean1 - 3 * popt[2], mean2 - 3 * popt[5])
        MAX = round(max(x),3) #max(mean1 + 3 * popt[2], mean2 + 3 * popt[5])
    
        np_mean1, np_mean2 = np.arange(MIN, MAX, step),np.arange(MIN, MAX, step)
        dim = len(np_mean1)

        np_R2, np_RMSQ = np.zeros([dim,dim]),np.zeros([dim,dim])
        check_R2_list,check_mean_list = [],[]
        for i, mean1 in enumerate(np_mean1):
            for j, mean2 in enumerate(np_mean2):
                if i < j:
                    try:
                        R2,popt,RMSQ,bounds = test_r2(x, y, mean1, mean2, step, MIN,MAX)
                        R2 = 0 if R2 < 0 else R2
                        np_R2[i][j] = R2
                        np_R2[j][i] = R2
                        np_RMSQ[i][j] = RMSQ
                        np_RMSQ[j][i] = RMSQ
                        check_R2_list.append(R2)
                        check_mean_list.append([mean1,mean2])
                    except Exception as e:
                        print(f"An error occurred: {e}")
                        np_R2[i][j] = 0
                        np_R2[j][i] = 0
                        np_RMSQ[i][j] = max(np_RMSQ.flat)
                        np_RMSQ[j][i] = max(np_RMSQ.flat)
                if i == j:
                    try:
                        R2,popt,RMSQ,bounds = test_r2(x, y, mean1, mean2, step, MIN,MAX)
                        R2 = 0 if R2 < 0 else R2
                        np_R2[i][j] = R2
                        np_RMSQ[i][j] = RMSQ
                        check_R2_list.append(R2)
                        check_mean_list.append([mean1,mean2])
                    except Exception as e:
                        print(f"An error occurred: {e}")
                        np_R2[i][j] = 0
                        np_RMSQ[i][j] = max(np_RMSQ.flat)  
        best_mean_combo = [j for i,j in zip(check_R2_list, check_mean_list) if i == max(check_R2_list)][0]
        mean1, mean2 = best_mean_combo[0], best_mean_combo[1]
        R2,popt,RMSQ,bounds = test_r2(x,y,mean1, mean2, step, MIN,MAX)
        
        #the fifth plot at the top right, the R2 score info.
        ax5 = fig.add_subplot(gs[0:2, 2:4])
        ax5.contourf(np_mean1, np_mean2, np_R2,20, cmap="plasma")
        ax5.set_title("The R2 Scores of the mean values.", size = 16)
        plt.xlabel("Mean1 (Å)", size = 12, labelpad = 2)
        plt.ylabel("Mean2 (Å)", size = 12, labelpad = 2)

        #the sixth plot at the left bottom, the RMSQ info.
        ax6 = fig.add_subplot(gs[2:4, 0:2])
        ax6.contourf(np_mean1, np_mean2, np_RMSQ,20, cmap="plasma")
        ax6.set_title("The RMSQ of the mean values.", size = 16)
        plt.xlabel("Mean1 (Å)", size = 12, labelpad = 2)
        plt.ylabel("Mean2 (Å)", size = 12, labelpad = 2)

        #the seventh plot at the bottom right, the fitting plots.
        x_perf = np.arange(MIN,MAX, 0.001)
        y_perf = [gaussian(X, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]) for X in x_perf]
        x_orig = [sorted_xy[l][0] for l in range(len(temp_xy)) if MIN<sorted_xy[l][0]<MAX]
        y_orig = [sorted_xy[l][1] for l in range(len(temp_xy)) if MIN<sorted_xy[l][0]<MAX]
        ax7 = plt.subplot(gs[2:4, 2:4])
        ori, = ax7.plot(x_orig,y_orig)
        gau, = ax7.plot(x_perf,y_perf, c="red")
        ax7.legend(handles=[ori,gau],labels=['Original_Line','Gaussian_Fitting_line'],loc = "upper right", fontsize = 16)
        ax7.set_title("The Fitting Curve and the Original Distribution.", size = 16)
        ax7.set_xlim([min(np_mean1),max(np_mean2)])
        plt.xlabel("Distance (Å)", size = 12, labelpad = 4)
        plt.ylabel("Proportion (%)", size = 12, labelpad = 4)
        threshold = round(max(popt[1] + 4 * popt[2], popt[4] + 4 * popt[5]),3)
        plt.suptitle("The Threshold between %s and %s should be %sÅ."%(ele1, ele2,threshold), size = 36)
        fig.text(s='mean_1 = %s, sigma_1 = %s, mean_2 = %s, sigma_2 = %s'%(round(popt[1],3),round(popt[2],3),round(popt[4],3),round(popt[5],3)),
                    x=0.5, y=0.92, fontsize=12, ha='center', va='center')
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.savefig(save_path + "Threshold of %s and %s.svg"%(ele1,ele2))
        parameters = [(ele1, ele2), {"R2":max(check_R2_list), "mean_1":round(popt[1],3), "sigma_1": round(popt[2],3), 
                                                              "mean_2":round(popt[4],3), "sigma_2": round(popt[5],3),
                                                              "threshold":threshold}]
        return parameters




"""
def refine_data(Min, Max, step, target_list):
    stations = [round(i,3) for i in np.arange(Min, Max, step)]
    stations.append(round(Max,3))
    
    processed_list = []
    for i in target_list:
        for j in stations:
            if j <= i and i < j + step:
                i = j
                processed_list.append(i)    
    return processed_list
"""
class GaussianModel:
    def __init__(self, num_gaussians):
        self.num_gaussians = num_gaussians  # Number of Gaussian functions to use

    def gaussian(self, x, *params):
        """
        Sum of multiple Gaussian functions.
        Params should be a flat list: [ampl1, mean1, stddev1, ampl2, mean2, stddev2, ...]
        """
        y = np.zeros_like(x)
        for i in range(self.num_gaussians):
            amplitude = params[i*3]
            mean = params[i*3+1]
            stddev = params[i*3+2]
            y += amplitude * np.exp(-((x - mean) ** 2) / (2 * (stddev ** 2)))
        return y

    def fit(self, x, y, means, step, MIN, MAX):
        """
        Fit the sum of Gaussians to the data, automatically generating initial guesses and bounds.
        """
        initial_guesses = []
        lower_bounds = []
        upper_bounds = []
        for mean in means:
            amplitude_guess = max(y) / self.num_gaussians
            stddev_guess = (MAX - MIN) / (2 * self.num_gaussians)  # Sensible default for stddev
            
            initial_guesses.extend([amplitude_guess, mean, stddev_guess])
            lower_bounds.extend([0, mean - 0.1 * step, 0])
            upper_bounds.extend([1, mean + 0.1 * step, (MAX - MIN) / 8])

        bounds = (lower_bounds, upper_bounds)
        popt, _ = curve_fit(self.gaussian, x, y,p0=initial_guesses, bounds=bounds)
        y_pred = [self.gaussian(X, *popt) for X in x]
        R2 = r2_score(y, y_pred)
        RMSQ = mean_squared_error(y, y_pred, squared=False)
        return R2, popt, RMSQ, bounds