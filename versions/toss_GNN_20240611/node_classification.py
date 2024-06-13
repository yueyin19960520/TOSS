import pickle
import time
import random
import shutil
from model_utils_dgl import *
from model_utils_pyg import *
from dataset_utils_dgl import *
from dataset_utils_pyg import *
from data_utils import *
import torch
from torch.utils.data import DataLoader as DGL_DataLoader
from torch_geometric.loader import DataLoader as PYG_DataLoader
import numpy as np
from torch.nn import BCEWithLogitsLoss,BCELoss
import os
import argparse


#################################################################################################################
################################################# dgl functions #################################################
#################################################################################################################
def dgl_train(model, data_loader, loss_func):
    global optimizer
    model.train()
    train_matrix = measure_matrix()

    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, masks = batch_data
        bg = bg.to('cuda:0')
        labels = labels.to('cuda:0')
        masks = masks.to('cuda:0')
        
        atom_feats = bg.ndata.pop('h')
        bond_feats = bg.edata.pop('e')
        atom_feats = atom_feats.to("cuda:0")
        bond_feats = bond_feats.to("cuda:0")

        model = model.to("cuda:0")
        try:
            outputs = model(bg, atom_feats, bond_feats)
        except:
            outputs = model(bg, atom_feats)
        outputs = outputs.to("cuda:0") #without sigmoid!

        #loss = (loss_func(outputs, labels) * (masks != 0).float()).mean()
        loss = loss_func(outputs, labels).mean()

        loss = loss.to("cuda:0")

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()
        train_matrix.update(outputs, labels, masks)

    try:
        roc_score = np.mean(train_matrix.roc_auc_score())
    except:
        roc_score = 0
    try:
        prc_score = np.mean(train_matrix.roc_precision_recall_score()) 
    except:
        prc_score = 0
    abs_score = train_matrix.absolute_correct_rate() *  100

    return roc_score, prc_score, abs_score, loss
        

def dgl_evaluate(model, data_loader):
    model.eval()
    eval_matrix = measure_matrix()

    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            bg = bg.to('cuda:0')
            labels = labels.to('cuda:0')
            masks = masks.to('cuda:0')
                
            atom_feats = bg.ndata.pop('h')
            bond_feats = bg.edata.pop('e')
            atom_feats = atom_feats.to("cuda:0")
            bond_feats = bond_feats.to("cuda:0")

            model = model.to("cuda:0")
            try:
                outputs = model(bg, atom_feats, bond_feats)
            except:
                outputs = model(bg, atom_feats)
            outputs = outputs.to("cuda:0") #without sigmoid!
                
            torch.cuda.empty_cache()
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

        loss = loss_func(outputs, labels).mean()

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
    abs_score = train_matrix.absolute_correct_rate() *  100

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



################################################################################################################
################################################# SAME FOR ALL #################################################
################################################################################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Choose the package use and the model.')
    parser.add_argument('-p', '--package', type=str, required=True, help='package use (PyG or DGL)')
    parser.add_argument('-m', '--model', type=str, required=True, help='model name (like GCN)')
    parser.add_argument('-s', '--size', type=str, required=True, help='model size (s means large model; w means small mode)')
    args = parser.parse_args()

    seed = 42
    setup_seed(42)
    path = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]

    model_path = os.path.join(path, "models")
    os.mkdir(model_path) if not os.path.exists(model_path) else None

    raw_path = os.path.join(path, "raw")
    os.mkdir(raw_path) if not os.path.exists(raw_path) else None
    shutil.copy(os.path.join(path, "graphs_dict.pkl"), os.path.join(raw_path, "graphs_dict.pkl"))

    processed_path = os.path.join(path, "processed")
    os.mkdir(processed_path) if not os.path.exists(processed_path) else None

    package_use = args.package

    if package_use.lower() == "dgl":
        dataset_path = os.path.join(raw_path, "NN_NC_DGL_dataset.pkl")
        if os.path.exists(dataset_path):
            print("Getting processed dataset...")
            file_get= open(dataset_path,"rb")
            tr_dataset, te_dataset, vl_dataset = pickle.load(file_get)
            file_get.close()
            print("Done!")
        else:
            print("Getting the all datapoint from the TOSS result dictonary...")
            file_get= open(os.path.join(raw_path,"graphs_dict.pkl"),"rb")
            graphs_dict = pickle.load(file_get)
            file_get.close()
            print("Done!")

            print("Splitting the original dataset to train, validate, and test dataset...")
            tr_graphs, vl_graphs, te_graphs = split_dataset_to_three(graphs_dict,seed,ratio=[3,1,1])
            print("Done!")

            print("Constructing the three graph_dicts for Link Prediction...")
            print("Refining the original graph_dicts...")
            tr_graphs = refine_graphs_dict(tr_graphs, criterion = ["NOSELF"])
            te_graphs = refine_graphs_dict(te_graphs, criterion = ["NOSELF"])
            vl_graphs = refine_graphs_dict(vl_graphs, criterion = ["NOSELF"])
            print("The original total graphs is %s, now the graphs is %s"%(len(graphs_dict),len(tr_graphs)+len(te_graphs)+len(vl_graphs)))
            print("Done!")

            print("Preparing the datasets...")
            tr_dataset = TOSS_DGL_NC_MEF_DataSet(tr_graphs)
            te_dataset = TOSS_DGL_NC_MEF_DataSet(te_graphs)
            vl_dataset = TOSS_DGL_NC_MEF_DataSet(vl_graphs)
            print("Done!")
    
            print("Saving the three datasets...")
            file_save= open(dataset_path,"wb")
            pickle.dump((tr_dataset, te_dataset, vl_dataset), file_save)
            file_save.close()
            print("Done!")
        
            print("All preparations are done!")

        tr_loader = DGL_DataLoader(tr_dataset, batch_size=100, shuffle=True, collate_fn=DGL_NC_collate, drop_last = True)
        vl_loader = DGL_DataLoader(vl_dataset, batch_size=100, shuffle=True, collate_fn=DGL_NC_collate, drop_last = True)
        te_loader = DGL_DataLoader(te_dataset, batch_size=100, shuffle=True, collate_fn=DGL_NC_collate, drop_last = True)


        dgl_GCN_w = dgl_GCNPredictor(in_feats=15, hidden_feats=[64, 64], 
                                     gnn_norm=None, activation=None, residual=None, batchnorm=None, dropout=None, 
                                     predictor_hidden_feats=64, n_tasks=12, predictor_dropout=0.3) 

        dgl_GCN_s = dgl_GCNPredictor(in_feats=15, hidden_feats=[256, 256, 256, 256], 
                                     gnn_norm=None, activation=None, residual=None, batchnorm=None, dropout=None, 
                                     predictor_hidden_feats=64, n_tasks=12, predictor_dropout=0.3) 

        dgl_GAT_w = dgl_GATPredictor(in_feats=15, hidden_feats=[16, 16], num_heads=[4, 4], 
                                     feat_drops=None, attn_drops=None, alphas=None, residuals=None, agg_modes=None, 
                                     activations=None, biases=None, 
                                     predictor_hidden_feats=64, n_tasks=12, predictor_dropout=0.3) 

        dgl_GAT_s = dgl_GATPredictor(in_feats=15, hidden_feats=[64, 64, 64, 64], num_heads=[4, 4, 4, 4], 
                                     feat_drops=None, attn_drops=None, alphas=None, residuals=None, agg_modes=None, 
                                     activations=None, biases=None, 
                                     predictor_hidden_feats=64, n_tasks=12, predictor_dropout=0.3)                           

        dgl_AFP_w = dgl_AFPPredictor(node_feat_size=15, edge_feat_size=13, dropout=0., 
                                     num_layers=2, graph_feat_size=64,
                                     predictor_hidden_feats=64, n_tasks=12, predictor_dropout=0.3)

        dgl_AFP_s = dgl_AFPPredictor(node_feat_size=15, edge_feat_size=13, dropout=0., 
                                     num_layers=4, graph_feat_size=256,
                                     predictor_hidden_feats=64, n_tasks=12, predictor_dropout=0.1)

        dgl_MPNN_w = dgl_MPNNPredictor(node_in_feats=15, edge_in_feats=13, 
                                       num_step_message_passing=2, node_out_feats=8, edge_hidden_feats=64,
                                       predictor_hidden_feats=64, n_tasks=12, predictor_dropout=0.3) 
 
        dgl_MPNN_s = dgl_MPNNPredictor(node_in_feats=15, edge_in_feats=13, 
                                       num_step_message_passing=4, node_out_feats=16, edge_hidden_feats=256,
                                       predictor_hidden_feats=64, n_tasks=12, predictor_dropout=0.3)

        train, evaluate = dgl_train, dgl_evaluate

    if package_use.lower() == "pyg":
        print("Getting the processed datasets...")
        dataset = TOSS_PYG_NC_MEF_DataSet(root=path)
        dataset = dataset.shuffle()  # shuffle the dataset
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

        pyg_GCN_w = pyg_GCNPredictor(in_feats=15, hidden_feats=[64, 64], 
                                     activation=None, residual=None, batchnorm=None, dropout=None, 
                                     predictor_hidden_feats=64, n_tasks=12, predictor_dropout=0.3) 

        pyg_GCN_s = pyg_GCNPredictor(in_feats=15, hidden_feats=[256, 256, 256, 256], 
                                     activation=None, residual=None, batchnorm=None, dropout=None, 
                                     predictor_hidden_feats=64, n_tasks=12, predictor_dropout=0.3) 

        pyg_GAT_w = pyg_GATPredictor(in_feats=15, hidden_feats=[16, 16], num_heads=[4, 4], 
                                     dropouts=None, biases=None, alphas=None, activations=None, agg_modes=None, 
                                     predictor_hidden_feats=64, n_tasks=12, predictor_dropout=0.3)

        pyg_GAT_s = pyg_GATPredictor(in_feats=15, hidden_feats=[64, 64, 64, 64], num_heads=[4, 4, 4, 4], 
                                     dropouts=None, biases=None, alphas=None, activations=None, agg_modes=None, 
                                     predictor_hidden_feats=64, n_tasks=12, predictor_dropout=0.3)

        pyg_AFP_w = pyg_AFPPredictor(node_feat_size=15, edge_feat_size=13, dropout=0.,
                                     num_layers=2, graph_feat_size=64, 
                                     predictor_hidden_feats=64, n_tasks=12, predictor_dropout=0.3)

        pyg_AFP_s = pyg_AFPPredictor(node_feat_size=15, edge_feat_size=13, dropout=0.,
                                     num_layers=4, graph_feat_size=256, 
                                     predictor_hidden_feats=64, n_tasks=12, predictor_dropout=0.3)

        pyg_MPNN_w= PyG_MPNNPredictor(node_in_feats=15, edge_in_feats=13, node_out_feats=8, edge_hidden_feats=64, 
                                     num_step_message_passing=2,
                                     n_tasks=12, predictor_hidden_feats=64, predictor_dropout=0.3)

        pyg_MPNN_s= PyG_MPNNPredictor(node_in_feats=15, edge_in_feats=13, node_out_feats=16, edge_hidden_feats=256, 
                                     num_step_message_passing=4,
                                     n_tasks=12, predictor_hidden_feats=64, predictor_dropout=0.3)

        train, evaluate = pyg_train, pyg_evaluate

    ##############################################################################################################################
    ######################################## SAME FOR BOTH PYG/DGL and ALL MODELS ################################################
    ##############################################################################################################################
    model_name = args.package.lower() + "_" + args.model.upper() + "_" + args.size.lower()
    exec("model = %s"%model_name)

    date = "%02d"%time.localtime()[1] + "%02d"%time.localtime()[2] 

    epochs = 200
    loss_func = BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=10 ** -3, weight_decay=10 ** -4)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4,16,48,80,100,120], gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

    #f_name = "/%s_%s.txt"%(model_name, date)
    f_name = os.path.join(model_path, "%s_%s.txt"%(model_name, date))
    m_name = os.path.join(model_path, "%s_%s.pth"%(model_name, date))

    print("Model Trainning...")
    parameter_space = sum(list(map(lambda x:model.state_dict()[x].reshape(-1,1).shape[0],list(model.state_dict().keys()))))
    print("The total number of parameters of the Model is %s"%parameter_space)
    
    best, patience = 0,0
    for epoch in range(epochs):
        s_time = time.time()

        tr_roc, tr_prc, tr_abs, loss = train(model, tr_loader, loss_func)
        tr_roc, tr_prc, tr_abs = "%.6f"%tr_roc, "%.6f"%tr_prc, "%.2f"%tr_abs

        vl_roc, vl_prc, vl_abs = evaluate(model, vl_loader)
        vl_roc, vl_prc, vl_abs = "%.6f"%vl_roc, "%.6f"%vl_prc, "%.2f"%vl_abs

        te_roc, te_prc, te_abs = evaluate(model, te_loader)
        te_roc, te_prc, te_abs = "%.6f"%te_roc, "%.6f"%te_prc, "%.2f"%te_abs

        e_time = time.time()
        cost = e_time-s_time
        cost = "%.2f"%cost
        scheduler.step()
        
        print("Epoch:%s,Cost:%ss. Train_Rate:%s, Validation_Rate:%s, Test_Rate:%s."%(epoch,cost,tr_abs,vl_abs,te_abs))
        with open(file=f_name,mode="a",encoding="utf-8") as f:
            data=f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"
                %(epoch,cost,tr_roc,tr_prc,tr_abs,vl_roc,vl_prc,vl_abs,te_roc,te_prc,te_abs))

        learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        best, patience, stop_flag, save_flag = stopper(vl_abs, best, epoch, patience, lr=learning_rate, max_patience=20, tolerance=0.01)

        if save_flag:
            torch.save(model.state_dict(), m_name)
            print("saved!")
        else:
            None
        if stop_flag:
            print("break")
            break
        else:
            print("continue...")
            continue
######################################################################################
"""END HERE"""