import pickle
import time
import random
from model_utils_pyg import *
from dataset_utils_pyg import *
from data_utils import *
import torch
from torch_geometric.loader import DataLoader as PYG_DataLoader
import numpy as np
from torch.nn import BCEWithLogitsLoss,BCELoss,MSELoss
import os
import argparse


def train(model, data_loader, loss_func):
    model.train()
    train_matrix = measure_matrix()

    for batch_id, batch_data in enumerate(data_loader):

        batch_data = batch_data.to('cuda:0')
        labels = batch_data.y.to('cuda:0')

        model = model.to("cuda:0")

        outputs = model(batch_data)

        outputs = outputs.to("cuda:0")

        loss = loss_func(outputs, labels)

        loss = loss.to("cuda:0")

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()

        masks = torch.ones(labels.shape)
        train_matrix.update(outputs, labels, masks)

    abs_rate = train_matrix.regression_absolute_rate()
    bvs_mae, bvs_mse = train_matrix.BVS_MASE()
    return bvs_mae, bvs_mse, abs_rate, loss


def evaluate(model, data_loader):
    model.eval()
    eval_matrix = measure_matrix()

    for batch_id, batch_data in enumerate(data_loader):

        batch_data = batch_data.to('cuda:0')
        labels = batch_data.y.to('cuda:0')

        model = model.to("cuda:0")

        outputs = model(batch_data)

        outputs = outputs.to("cuda:0")

        torch.cuda.empty_cache()

        masks = torch.ones(labels.shape)
        eval_matrix.update(outputs, labels, masks)

    abs_rate = eval_matrix.regression_absolute_rate()
    bvs_mae, bvs_mse = eval_matrix.BVS_MASE()
    return bvs_mae, bvs_mse, abs_rate


#########################################################################################
####################################### MAIN PROCESS ####################################
#########################################################################################
if __name__ == "__main__":
    seed = 42
    setup_seed(42)
    path = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
    model_path = os.path.join(path, "models")

    print("Getting the processed datasets...")
    dataset = TOSS_PYG_BVR_MEF_DataSet(root=path)
    dataset = dataset.shuffle()  # shuffle the dataset
    print("Done!")
        
    print("Splitting the original dataset to train, validate, and test dataset...")
    tr_dataset = dataset[:int(len(dataset)*0.8)]  # 80% of data for training
    vl_dataset = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]  # 10% of data for validation
    te_dataset = dataset[int(len(dataset)*0.9):] 
    print("Done!")
    print("All preparations are done!")

    PYG_BVR_collate = PYG_NC_collate
    tr_loader = PYG_DataLoader(tr_dataset, batch_size=100, shuffle=True, drop_last=True, collate_fn=PYG_BVR_collate)  
    vl_loader = PYG_DataLoader(vl_dataset, batch_size=100, shuffle=True, drop_last=True, collate_fn=PYG_BVR_collate)  
    te_loader = PYG_DataLoader(te_dataset, batch_size=100, shuffle=True, drop_last=True, collate_fn=PYG_BVR_collate) 


    pyg_BVR_GCN_w = pyg_GCNBondValencePredictor(in_feats=15, hidden_feats=[256,256,256,256], 
                                                activation=None, residual=None, batchnorm=None, dropout=None, 
                                                predictor_hidden_feats=64, predictor_dropout=0.3) 

    pyg_BVR_GCN_s = pyg_GCNBondValencePredictor(in_feats=15, hidden_feats=[512,512,512,512], 
                                                activation=None, residual=None, batchnorm=None, dropout=None, 
                                                predictor_hidden_feats=128, predictor_dropout=0.3)

    model_name = "pyg_BVR_GCN_s"
    exec("model = %s"%model_name)
    date = "%02d"%time.localtime()[1] + "%02d"%time.localtime()[2] 

    loss_func = MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=10 ** -3, weight_decay=10 ** -4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4, eps=1e-8)

    f_name = os.path.join(model_path, "%s_%s.txt"%(model_name, date))
    m_name = os.path.join(model_path, "%s_%s.pth"%(model_name, date))

    print("Model Trainning...")
    parameter_space = sum(list(map(lambda x:model.state_dict()[x].reshape(-1,1).shape[0],list(model.state_dict().keys()))))
    print("The total number of parameters of the Model is %s"%parameter_space)
    epochs = 300
    best, patience = 0,0
    for epoch in range(epochs):
        s_time = time.time()

        tr_mae, tr_mse, tr_abs, loss = train(model, tr_loader, loss_func)
        tr_mae, tr_mse, tr_abs = "%.4f"%tr_mae, "%.4f"%tr_mse, "%.4f"%tr_abs

        vl_mae, vl_mse, vl_abs = evaluate(model, vl_loader)
        vl_mae, vl_mse, vl_abs = "%.4f"%vl_mae, "%.4f"%vl_mse, "%.4f"%vl_abs

        te_mae, te_mse, te_abs = evaluate(model, te_loader)
        te_mae, te_mse, te_abs = "%.4f"%te_mae, "%.4f"%te_mse, "%.4f"%te_abs

        e_time = time.time()
        cost = e_time-s_time
        cost = "%.2f"%cost
        scheduler.step(loss)
        
        print("Epoch:%s,Cost:%ss. Train_MAE:%s, Train_MSE:%s, Train_ABS:%s, Validation_MAE:%s, Validation_MSE:%s, Validation_ABS:%s, Test_MAE:%s, Test_MSE:%s, Test_ABS:%s."
        	%(epoch,cost,tr_mae,tr_mse,tr_abs,vl_mae,vl_mse,vl_abs,te_mae,te_mse,te_abs))
        with open(file=f_name,mode="a",encoding="utf-8") as f:
            data=f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"
                %(epoch,cost,tr_mae,tr_mse,tr_abs,vl_mae,vl_mse,vl_abs,te_mae,te_mse,te_abs))

        learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        best, patience, stop_flag, save_flag = stopper(vl_mae, best, epoch, patience, lr=learning_rate, max_patience=20, tolerance=0.01)

        if save_flag:
            torch.save(model.state_dict(), m_path)
            print("saved!")
        else:
            None
        if stop_flag:
            print("break")
            break
        else:
            print("continue...")
            continue

