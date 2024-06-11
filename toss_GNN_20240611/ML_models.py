import pickle
import pandas as pd
import numpy as np
import random
import os
import joblib

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve

#################################################################################################################
################################################PRESET PARAMETERS################################################
#################################################################################################################

def standardize(col):
    return (col - np.mean(col)) / np.std(col)

def absolute_correct_rate(y_pred, y_true):
        score = 0
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        for i in range(y_pred.shape[0]):
            t = np.where(max(y_true[i]) == y_true[i], 1, 0)
            p = np.where(max(y_pred[i]) == y_pred[i], 1, 0)
            if (t==p).all():
                score += 1
        return score/y_pred.shape[0]

################################################################################################################
###############################################SAME FOR ALL MODEL###############################################
################################################################################################################
seed = 42
path = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
model_path = path + "/models"

try:
    print("Getting pre_load graph dicts...")
    file_get = open(path + "/ML_NC_split_dataset.pkl","rb")
    (data_tr_y, data_tr_x, data_vl_y, data_vl_x, data_te_y, data_te_x) = pickle.load(file_get)
    file_get.close()
    print("Done!")
except:
    print("Failed of getting pre_load graph dicts...")
    print("Getting the all datapoint from the TOSS result dictonary...")
    file_get= open(path + "/graphs_dict.pkl","rb")
    graphs_dict = pickle.load(file_get)
    file_get.close()
    print("Done!")
    
    print("Preparing the datasets...")
    columns = graphs_dict["mp-31770.cif"]["n"].columns.to_list()
    all_data = np.vstack(list(map(lambda x:x["n"], list(graphs_dict.values()))))
    ML_data = pd.DataFrame(all_data, columns=columns)
    #ML_data.to_csv(path + "/ML_data.csv")

    ML_data[columns[0:-1]] = ML_data[columns[0:-1]].apply(standardize, axis=0)

    all_idx = np.arange(ML_data.shape[0])
    random.shuffle(all_idx)
    N = len(all_idx)
    tr_idx = all_idx[0:3*N//5]
    vl_idx = all_idx[3*N//5:4*N//5]
    te_idx = all_idx[4*N//5:]

    data_tr = ML_data.iloc[tr_idx,:].reset_index(drop=True)
    data_vl = ML_data.iloc[vl_idx,:].reset_index(drop=True)
    data_te = ML_data.iloc[te_idx,:].reset_index(drop=True)

    data_tr_y = data_tr["OS"].values.reshape(-1, 1).ravel()
    data_tr_x = np.array(data_tr.iloc[:,:-1].values)
    
    data_vl_y = data_vl["OS"].values.reshape(-1, 1).ravel()
    data_vl_x = np.array(data_vl.iloc[:,:-1].values)
    
    data_te_y = data_te["OS"].values.reshape(-1, 1).ravel()
    data_te_x = np.array(data_te.iloc[:,:-1].values)
    print("Done!")


    print("Saving the prepared datasets...")
    file_save= open(path + "/ML_NC_split_dataset.pkl","wb")
    pickle.dump((data_tr_y, data_tr_x, data_vl_y, data_vl_x, data_te_y, data_te_x), file_save)
    file_save.close()
    print("Done!")


if __name__ == "__main__":


    RF_2000 = RandomForestClassifier(n_estimators=2000, max_depth=25, min_impurity_decrease=0, criterion='entropy',
                                     min_samples_leaf=50,min_samples_split=50, max_leaf_nodes=None,
                                     n_jobs=48, random_state=1, verbose=0, class_weight='balanced')

    RF_200 = RandomForestClassifier(n_estimators=200, max_depth=25, min_impurity_decrease=0, criterion='entropy',
                                    min_samples_leaf=50,min_samples_split=50, max_leaf_nodes=None,
                                    n_jobs=48, random_state=1, verbose=0, class_weight='balanced')

    XGB_2000 = XGBClassifier(booster="gbtree", verbosity=0, n_jobs=48,
                             n_estimators=2000, max_depth=20, min_child_weight=1, subsample=1, colsample_bytree=1, 
                             learning_rate=0.01, gamma=0, random_state=1)

    XGB_200 = XGBClassifier(booster="gbtree", verbosity=0, n_jobs=48,
                            n_estimators=200, max_depth=20, min_child_weight=1, subsample=1, colsample_bytree=1, 
                            learning_rate=0.01, gamma=0, random_state=1)

    le = LabelEncoder()
    oe = OneHotEncoder()

    model_name = str(input("Give the name of the model from 'RF_200, RF_2000, XGB_200, XGB_2000':"))

    exec("model = %s"%model_name)

    print("Model training...")
    model.fit(data_tr_x, le.fit_transform(data_tr_y))

    try:
        model.save_model(model_path + "/%s.json"%model_name)
    except:
        joblib.dump(model, model_path+"/%s.json"%model_name)

    vl_preds = model.predict_proba(data_vl_x)
    vl_labels = oe.fit_transform(data_vl_y.reshape(-1,1)).toarray() #one_hot encode!!
    vl_roc = roc_auc_score(vl_labels, vl_preds, multi_class="ovo")
    vl_score = absolute_correct_rate(vl_preds, vl_labels)

    te_preds = model.predict_proba(data_te_x)
    te_labels = oe.fit_transform(data_te_y.reshape(-1,1)).toarray() #one_hot encode!!
    te_roc = roc_auc_score(te_labels, te_preds, multi_class="ovo")
    te_score = absolute_correct_rate(te_preds, te_labels)

    print("Validate ROC:%s, Validate Rate:%s, Test ROC:%s, Validate Rate:%s."%(vl_roc, vl_score, te_roc, te_score))


    data_all_x = np.vstack((data_te_x, data_vl_x, data_tr_x))
    data_all_y = np.hstack((data_te_y, data_vl_y, data_tr_y))

    all_preds = model.predict_proba(data_all_x)
    all_labels = oe.fit_transform(data_all_y.reshape(-1,1)).toarray()
    all_roc = roc_auc_score(all_labels, all_preds, multi_class="ovo")
    all_score = absolute_correct_rate(all_preds, all_labels)

    print("ALL data: ROC:%s, Correct Rate:%s."%(all_roc, all_score))
"""END HERE"""