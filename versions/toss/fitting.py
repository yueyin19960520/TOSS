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


def gaussian(x, amplified1, mean1, stddev1, amplified2, mean2, stddev2):
    return amplified1 * np.exp((-(x-mean1)**2)/(2*(stddev1**2))) + amplified2 * np.exp((-(x-mean2)**2)/(2*(stddev2**2)))


def test_r2(x, y, mean1, mean2,step, MIN,MAX):
    bounds = ([0,mean1-0.1*step, 0, 0,mean2-0.1*step, 0], [0.2,mean1+0.1*step,(MAX-MIN)/8,1 ,mean2+0.1*step,(MAX-MIN)/8])
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
    stations = [round(i,3) for i in np.arange(Min, Max, step)]
    stations.append(round(Max,3))
    
    processed_list = []
    for i in target_list:
        for j in stations:
            if j <= i and i < j + step:
                i = j
                processed_list.append(i)    
    return processed_list


def get_bond_length_distribution_fitting_info(ele1, ele2, former_threshold, refined_data, global_save_path):
    #plot the global bong length and its distribuion.
    path_for_all = global_save_path
    try:
        global_data = pd.read_csv(path_for_all + "bond_of_%s_%s.csv"%(ele1,ele2), header = None)
    except:
        try:
            global_data = pd.read_csv(path_for_all + "bond_of_%s_%s.csv"%(ele2,ele1), header = None)
        except:
            global_data = []
            refined_data = []

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
                    except:
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
                    except:
                        np_R2[i][j] = 0
                        np_RMSQ[i][j] = max(np_RMSQ.flat)  
        best_mean_combo = [j for i,j in zip(check_R2_list, check_mean_list) if i == max(check_R2_list)][0]
        mean1, mean2 = best_mean_combo[0], best_mean_combo[1]
        R2,popt,RMSQ,bounds = test_r2(x,y,mean1, mean2, step, MIN,MAX)
        
        #the seventh plot at the bottom right, the fitting plots.
        x_perf = np.arange(MIN,MAX, 0.001)
        y_perf = [gaussian(X, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]) for X in x_perf]
        x_orig = [sorted_xy[l][0] for l in range(len(temp_xy)) if MIN<sorted_xy[l][0]<MAX]
        y_orig = [sorted_xy[l][1] for l in range(len(temp_xy)) if MIN<sorted_xy[l][0]<MAX]

        threshold = round(max(popt[1] + 4 * popt[2], popt[4] + 4 * popt[5]),3)
        sigma_1, sigma_2 = round(popt[2],3), round(popt[5],3)
        mean1, mean2 = round(popt[1],3), round(popt[4],3)
        result = {"R2":round(R2,2), "mean_1":mean1, "sigma_1": sigma_1, "mean_2":mean2, "sigma_2":sigma_2,
                  "size":size, "threshold":threshold}
        parameters = [(ele1, ele2), result]
    else:
        parameters = [(ele1, ele2), "None"]
    return parameters


def bond_length_distribution_fitting(ele1, ele2, former_threshold, refined_data, save_path, global_save_path):
    #plot the global bong length and its distribuion.
    path_for_all = global_save_path
    try:
        global_data = pd.read_csv(path_for_all + "bond_of_%s_%s.csv"%(ele1,ele2), header = None)
    except:
        try:
            global_data = pd.read_csv(path_for_all + "bond_of_%s_%s.csv"%(ele2,ele1), header = None)
        except:
            global_data = []
            refined_data = []
            
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
                    except:
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
                    except:
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
"""END HERE"""
"""
def get_bond_length_distribution_fitting_info(ele1, ele2, former_threshold, refined_data):
    #plot the global bong length and its distribuion.
    path_for_all = "D:/share/ML for CoreLevel/redifined_bond_length/OQMD/global_length_csv/"
    try:
        global_data = pd.read_csv(path_for_all + "bond_of_%s_%s.csv"%(ele1,ele2), header = None)
    except:
        try:
            global_data = pd.read_csv(path_for_all + "bond_of_%s_%s.csv"%(ele2,ele1), header = None)
        except:
            global_data = []
            refined_data = []

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
                        R2,popt,RMSQ,bounds = test_r2(x, y, mean1, mean2, step)
                        R2 = 0 if R2 < 0 else R2
                        np_R2[i][j] = R2
                        np_R2[j][i] = R2
                        np_RMSQ[i][j] = RMSQ
                        np_RMSQ[j][i] = RMSQ
                        check_R2_list.append(R2)
                        check_mean_list.append([mean1,mean2])
                    except:
                        np_R2[i][j] = 0
                        np_R2[j][i] = 0
                        np_RMSQ[i][j] = max(np_RMSQ.flat)
                        np_RMSQ[j][i] = max(np_RMSQ.flat)
                if i == j:
                    try:
                        R2,popt,RMSQ,bounds = test_r2(x, y, mean1, mean2, step)
                        R2 = 0 if R2 < 0 else R2
                        np_R2[i][j] = R2
                        np_RMSQ[i][j] = RMSQ
                        check_R2_list.append(R2)
                        check_mean_list.append([mean1,mean2])
                    except:
                        np_R2[i][j] = 0
                        np_RMSQ[i][j] = max(np_RMSQ.flat)  
        best_mean_combo = [j for i,j in zip(check_R2_list, check_mean_list) if i == max(check_R2_list)][0]
        mean1, mean2 = best_mean_combo[0], best_mean_combo[1]
        R2,popt,RMSQ,bounds = test_r2(x,y,mean1, mean2, step)
        
        if popt[2] <= 0.014 or popt[5] <= 0.014:
            #CHANFE TO SINGLE FITTING
            np_R2, np_RMSQ = np.zeros([dim,dim]),np.ones([dim,dim])
            for i, mean1 in enumerate(np_mean1):
                try:
                    R2,popt,RMSQ,bounds = s_test_r2(x, y, mean1, step)
                    R2 = 0 if R2 < 0 else R2
                    np_R2[i][i] = R2
                    np_RMSQ[i][i] = RMSQ
                    check_R2_list.append(R2)
                    check_mean_list.append([mean1,mean2])
                except:
                    np_R2[i][i] = 0
                    np_RMSQ[i][i] = max(np_RMSQ.flat)
            best_mean_combo = [j for i,j in zip(check_R2_list, check_mean_list) if i == max(check_R2_list)][0]
            check_R2_list,check_mean_list = [],[]
            mean1 = round(best_mean_combo[0],3)
            R2,popt,RMSQ,bounds = s_test_r2(x,y,mean1, step)

            #the seventh plot at the bottom right, the fitting plots.
            x_perf = np.arange(MIN,MAX, 0.001)
            y_perf = [s_gaussian(X, popt[0], popt[1], popt[2]) for X in x_perf]
            x_orig = [sorted_xy[l][0] for l in range(len(temp_xy)) if MIN<sorted_xy[l][0]<MAX]
            y_orig = [sorted_xy[l][1] for l in range(len(temp_xy)) if MIN<sorted_xy[l][0]<MAX]

            threshold = round(mean1 + 3 * popt[2],3)
            sigma_1 = round(popt[2],3)
            result = {"R2":round(R2,2), "mean_1":mean1, "sigma_1": sigma_1,"size":size, "threshold":threshold}
        else:
            #the seventh plot at the bottom right, the fitting plots.
            x_perf = np.arange(MIN,MAX, 0.001)
            y_perf = [gaussian(X, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]) for X in x_perf]
            x_orig = [sorted_xy[l][0] for l in range(len(temp_xy)) if MIN<sorted_xy[l][0]<MAX]
            y_orig = [sorted_xy[l][1] for l in range(len(temp_xy)) if MIN<sorted_xy[l][0]<MAX]

            threshold = round(max(popt[1] + 3 * popt[2], popt[4] + 3 * popt[5]),3)
            sigma_1, sigma_2 = round(popt[2],3), round(popt[5],3)
            mean1, mean2 = round(popt[1],3), round(popt[4],3)
            result = {"R2":round(R2,2), "mean_1":mean1, "sigma_1": sigma_1, "mean_2":mean2, "sigma_2":sigma_2,
                  "size":size, "threshold":threshold}
        parameters = [(ele1, ele2), result]
    else:
        parameters = [(ele1, ele2), "None"]
    return parameters


def bond_length_distribution_fitting(ele1, ele2, former_threshold, refined_data):
    #plot the global bong length and its distribuion.
    path_for_all = "D:/share/ML for CoreLevel/redifined_bond_length/OQMD/global_length_csv/"
    try:
        global_data = pd.read_csv(path_for_all + "bond_of_%s_%s.csv"%(ele1,ele2), header = None)
    except:
        try:
            global_data = pd.read_csv(path_for_all + "bond_of_%s_%s.csv"%(ele2,ele1), header = None)
        except:
            global_data = []
            refined_data = []
            
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
                        R2,popt,RMSQ,bounds = test_r2(x, y, mean1, mean2, step)
                        R2 = 0 if R2 < 0 else R2
                        np_R2[i][j] = R2
                        np_R2[j][i] = R2
                        np_RMSQ[i][j] = RMSQ
                        np_RMSQ[j][i] = RMSQ
                        check_R2_list.append(R2)
                        check_mean_list.append([mean1,mean2])
                    except:
                        np_R2[i][j] = 0
                        np_R2[j][i] = 0
                        np_RMSQ[i][j] = max(np_RMSQ.flat)
                        np_RMSQ[j][i] = max(np_RMSQ.flat)
                if i == j:
                    try:
                        R2,popt,RMSQ,bounds = test_r2(x, y, mean1, mean2, step)
                        R2 = 0 if R2 < 0 else R2
                        np_R2[i][j] = R2
                        np_RMSQ[i][j] = RMSQ
                        check_R2_list.append(R2)
                        check_mean_list.append([mean1,mean2])
                    except:
                        np_R2[i][j] = 0
                        np_RMSQ[i][j] = max(np_RMSQ.flat)  
        best_mean_combo = [j for i,j in zip(check_R2_list, check_mean_list) if i == max(check_R2_list)][0]
        mean1, mean2 = best_mean_combo[0], best_mean_combo[1]
        R2,popt,RMSQ,bounds = test_r2(x,y,mean1, mean2, step)
        
        if popt[2] <= 0.014 or popt[5] <= 0.014:
            #CHANFE TO SINGLE FITTING
            np_R2, np_RMSQ = np.zeros([dim,dim]),np.ones([dim,dim])
            for i, mean1 in enumerate(np_mean1):
                try:
                    R2,popt,RMSQ,bounds = s_test_r2(x, y, mean1, step)
                    R2 = 0 if R2 < 0 else R2
                    np_R2[i][i] = R2
                    np_RMSQ[i][i] = RMSQ
                    check_R2_list.append(R2)
                    check_mean_list.append([mean1,mean2])
                except:
                    np_R2[i][i] = 0
                    np_RMSQ[i][i] = max(np_RMSQ.flat)
            best_mean_combo = [j for i,j in zip(check_R2_list, check_mean_list) if i == max(check_R2_list)][0]
            check_R2_list,check_mean_list = [],[]
            mean1 = round(best_mean_combo[0],3)
            R2,popt,RMSQ,bounds = s_test_r2(x,y,mean1, step)
            
            #the fifth plot at the top right, the R2 score info.
            ax5 = fig.add_subplot(gs[0:2, 2:4])
            ax5.contourf(np_mean1, np_mean1, np_R2,20, cmap="plasma")
            ax5.set_title("The R2 Scores of the mean values.", size = 16)
            plt.xlabel("Mean1 (Å)", size = 12, labelpad = 2)
            plt.ylabel("Mean1 (Å)", size = 12, labelpad = 2)

            #the sixth plot at the left bottom, the RMSQ info.
            ax6 = fig.add_subplot(gs[2:4, 0:2])
            ax6.contourf(np_mean1, np_mean1, np_RMSQ,20, cmap="plasma")
            ax6.set_title("The RMSQ of the mean values.", size = 16)
            plt.xlabel("Mean1 (Å)", size = 12, labelpad = 2)
            plt.ylabel("Mean1 (Å)", size = 12, labelpad = 2)

            #the seventh plot at the bottom right, the fitting plots.
            x_perf = np.arange(MIN,MAX, 0.001)
            y_perf = [s_gaussian(X, popt[0], popt[1], popt[2]) for X in x_perf]
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
            threshold = round(mean1 + 3 * popt[2],3)
            plt.suptitle("The Threshold between %s and %s should be %sÅ."%(ele1, ele2,threshold), size = 36)
            fig.text(s='mean_1 = %s, sigma_1 = %s'%(round(popt[1],3),round(popt[2],3)),
                     x=0.5, y=0.92, fontsize=12, ha='center', va='center')
            plt.subplots_adjust(wspace=0.4, hspace=0.4)
            plt.savefig("D:/share/ML for CoreLevel/redifined_bond_length/temp/test/refined_length_png/Threshold of %s and %s.png"%(ele1,ele2))
        else:
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
            threshold = round(max(popt[1] + 3 * popt[2], popt[4] + 3 * popt[5]),3)
            plt.suptitle("The Threshold between %s and %s should be %sÅ."%(ele1, ele2,threshold), size = 36)
            fig.text(s='mean_1 = %s, sigma_1 = %s, mean_2 = %s, sigma_2 = %s'%(round(popt[1],3),round(popt[2],3),round(popt[4],3),round(popt[5],3)),
                     x=0.5, y=0.92, fontsize=12, ha='center', va='center')
            plt.subplots_adjust(wspace=0.4, hspace=0.4)
            plt.savefig("D:/share/ML for CoreLevel/redifined_bond_length/temp/test/refined_length_png/Threshold of %s and %s.png"%(ele1,ele2))
            #plt.show()


def test_r2(x, y, mean1, mean2,step):
    bounds = ([0,mean1-0.1*step, 0, 0,mean2-0.1*step, 0], [0.2,mean1+0.1*step, 0.2,1,mean2+0.1*step, 0.5])
    popt, _ = curve_fit(gaussian, x, y, bounds = bounds)
    y_pred = [gaussian(X,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5]) for X in x]
    y_true = y
    R2 = r2_score(y_true,y_pred)
    RMSQ = mean_squared_error(y_true,y_pred, squared=False)
    return R2,popt,RMSQ,bounds
"""