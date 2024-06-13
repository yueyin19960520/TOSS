import pickle
import time
import random
from model_utils_pyg import *
from dataset_utils_pyg import *
from data_utils import *
import torch
from torch_geometric.loader import DataLoader as PYG_DataLoader
import numpy as np
import os
from torch.nn import BCEWithLogitsLoss,BCELoss



#################################################################################################################
################################################# pyg functions #################################################
#################################################################################################################
def pyg_train(model, data_loader, loss_func):
    global optimizer
    model.train()
    train_matrix = measure_matrix()

    for batch_id, batch_data in enumerate(data_loader):

        batch_data = batch_data.to('cuda:0')
        labels = batch_data.y.to('cuda:0')

        model = model.to("cuda:0")

        outputs = model(batch_data)

        outputs = outputs.to("cuda:0")

        loss = loss_func(outputs, labels)#.mean()

        loss = loss.to("cuda:0")

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()

        masks = torch.ones(labels.shape)
        train_matrix.update(outputs, labels, masks)

    try:
        roc_score = np.mean(train_matrix.roc_auc_score())
    except:
        roc_score = 0
    try:
        prc_score = np.mean(train_matrix.roc_precision_recall_score()) 
    except:
        prc_score = 0
    abs_score = 0.0 #train_matrix.absolute_correct_rate() *  100

    return roc_score, prc_score, abs_score, loss


def pyg_evaluate(model, data_loader):
    model.eval()
    eval_matrix = measure_matrix()

    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):

            batch_data = batch_data.to('cuda:0')
            labels = batch_data.y.to('cuda:0')

            model = model.to("cuda:0")

            outputs = model(batch_data)

            outputs = outputs.to("cuda:0") #without sigmoid!
                
            torch.cuda.empty_cache()
            masks = torch.ones(labels.shape)
            eval_matrix.update(outputs, labels, masks)
    try:
        roc_score = np.mean(eval_matrix.roc_auc_score())
    except:
        roc_score = 0
    try:
        prc_score = np.mean(eval_matrix.roc_precision_recall_score()) 
    except:
        prc_score = 0
    abs_score = eval_matrix.absolute_correct_rate() *  100
    
    return roc_score, prc_score, abs_score


if __name__ == "__main__":
    seed = 42
    setup_seed(42)
    path = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]

    print("Getting the processed datasets...")
    dataset = TOSS_PYG_LP_PRE_MEF_DataSet(root=path)
    dataset = dataset.shuffle() 
    print("Done!")
        
    print("Splitting the original dataset to train, validate, and test dataset...")
    tr_dataset = dataset[:int(len(dataset)*0.8)]  # 80% of data for training
    vl_dataset = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]  # 10% of data for validation
    te_dataset = dataset[int(len(dataset)*0.9):] 
    print("Done!")
    print("All preparations are done!")

    tr_loader = PYG_DataLoader(tr_dataset, batch_size=100, shuffle=True, drop_last=True, collate_fn=PYG_NC_collate)  
    vl_loader = PYG_DataLoader(vl_dataset, batch_size=100, shuffle=True, drop_last=True, collate_fn=PYG_NC_collate)  
    te_loader = PYG_DataLoader(te_dataset, batch_size=100, shuffle=True, drop_last=True, collate_fn=PYG_NC_collate) 


    pyg_Hetero_GCN = pyg_Hetero_GCNPredictor(atom_feats=13, bond_feats=13, hidden_feats=[256,256,256,256], 
                                             activation=None, residual=None, batchnorm=None, dropout=None,
                                             predictor_hidden_feats=64, n_tasks=2,predictor_dropout=0.3)

    model_name = "pyg_Hetero_GCN"
    exec("model = %s"%model_name)

    loss_func = BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=10 ** -3, weight_decay=10 ** -4)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4,16,48,80,100,120], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4, eps=1e-8)


    print("Model is training...")
    epochs = 300
    best, patience = 0,0
    for epoch in range(epochs):
        s_time = time.time()

        tr_roc, tr_prc, tr_abs, loss = pyg_train(model, tr_loader, loss_func)
        tr_roc, tr_prc, tr_abs = "%.6f"%tr_roc, "%.6f"%tr_prc, "%.2f"%tr_abs

        vl_roc, vl_prc, vl_abs = pyg_evaluate(model, vl_loader)
        vl_roc, vl_prc, vl_abs = "%.6f"%vl_roc, "%.6f"%vl_prc, "%.2f"%vl_abs

        te_roc, te_prc, te_abs = pyg_evaluate(model, te_loader)
        te_roc, te_prc, te_abs = "%.6f"%te_roc, "%.6f"%te_prc, "%.2f"%te_abs

        e_time = time.time()
        cost = e_time-s_time
        cost = "%.2f"%cost
        scheduler.step(loss)

        print("Epoch: %s, Cost: %s, tr_roc:%s, tr_prc:%s, tr_abs:%s, vl_roc:%s, vl_prc:%s, vl_abs:%s, te_roc:%s, te_prc:%s, te_abs:%s."
            %(epoch, cost, tr_roc, tr_prc, tr_abs, vl_roc, vl_prc, vl_abs, te_roc, te_prc, te_abs))

        learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        print(learning_rate)
"""END HERE"""
