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
def dgl_Homo_train(model, data_loader, loss_func):
    global optimizer
    model.train()
    train_matrix = measure_matrix()
    
    for batch_id, batch_data in enumerate(data_loader):
        mids, bg, edge_labels = batch_data
        edge_labels = edge_labels.to('cuda:0')
            
        edge_masks = torch.from_numpy(np.ones(edge_labels.shape))
        edge_masks = edge_masks.to('cuda:0')
            
        bg = bg.to('cuda:0')
            
        atom_feats = bg.ndata.pop('h')
        atom_feats = atom_feats.to('cuda:0')

        bond_feats = bg.edata.pop('e')
        bond_feats = bond_feats.to('cuda:0')

        model = model.to('cuda:0')

        edge_preds = model(bg, atom_feats, bond_feats)

        loss = (loss_func(edge_preds, edge_labels) * (edge_masks != 0).float()).mean()

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()
        train_matrix.update(edge_preds, edge_labels, edge_masks)
    
    roc_score = np.mean(train_matrix.roc_auc_score())
    prc_score = np.mean(train_matrix.roc_precision_recall_score()) 
    #abs_score = train_matrix.logistic_absolute_rate()*100
    abs_score = 0.00#train_matrix.absolute_correct_rate()* 100
    
    return roc_score, prc_score, abs_score


def dgl_Homo_evaluate(model, data_loader):
    model.eval()
    eval_matrix = measure_matrix()

    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            mids, bg, edge_labels = batch_data
            edge_labels = edge_labels.to('cuda:0')
            
            edge_masks = torch.from_numpy(np.ones(edge_labels.shape))
            edge_masks = edge_masks.to('cuda:0')
            
            bg = bg.to('cuda:0')
            
            atom_feats = bg.ndata.pop('h')
            atom_feats = atom_feats.to('cuda:0')

            bond_feats = bg.edata.pop('e')
            bond_feats = bond_feats.to('cuda:0')

            model = model.to('cuda:0')

            edge_preds = model(bg, atom_feats, bond_feats)
                
            torch.cuda.empty_cache()
            eval_matrix.update(edge_preds, edge_labels, edge_masks)
            
    roc_score = np.mean(eval_matrix.roc_auc_score())
    prc_score = np.mean(eval_matrix.roc_precision_recall_score()) 
    #abs_score = eval_matrix.logistic_absolute_rate()*100
    abs_score = eval_matrix.absolute_correct_rate()*100
 
    return roc_score, prc_score, abs_score 


def dgl_Hetero_train(model, data_loader, loss_func):
    global optimizer
    model.train()
    train_matrix = measure_matrix()
    
    for batch_id, batch_data in enumerate(data_loader):
        mids, bg, bonds_labels = batch_data
        bonds_labels = bonds_labels.to('cuda:0')
            
        bonds_masks = torch.from_numpy(np.ones(bonds_labels.shape))
        bonds_masks = bonds_masks.to('cuda:0')
            
        bg = bg.to('cuda:0')
            
        feats = bg.ndata.pop('h')
        atoms_feats = feats['atoms'].to('cuda:0')
        bonds_feats = feats['bonds'].to('cuda:0')

        model = model.to('cuda:0')

        bonds_preds = model(bg, {'atoms':atoms_feats, 'bonds':bonds_feats})

        #loss = (loss_func(bonds_preds, bonds_labels) * (bonds_masks != 0).float()).mean()
        loss = loss_func(bonds_preds, bonds_labels)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()
        train_matrix.update(bonds_preds, bonds_labels, bonds_masks)
    
    roc_score = np.mean(train_matrix.roc_auc_score())
    prc_score = np.mean(train_matrix.roc_precision_recall_score()) 
    #abs_score = train_matrix.logistic_absolute_rate()*100
    abs_score = 0.00#train_matrix.absolute_correct_rate()* 100
    
    return roc_score, prc_score, abs_score, loss


def dgl_Hetero_evaluate(model, data_loader):
    model.eval()
    eval_matrix = measure_matrix()

    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            mids, bg, bonds_labels = batch_data
            bonds_labels = bonds_labels.to('cuda:0')
            
            bonds_masks = torch.from_numpy(np.ones(bonds_labels.shape))
            bonds_masks = bonds_masks.to('cuda:0')
            
            bg = bg.to('cuda:0')
            
            feats = bg.ndata.pop('h')
            atoms_feats = feats['atoms'].to('cuda:0')
            bonds_feats = feats['bonds'].to('cuda:0')

            model = model.to('cuda:0')

            bonds_preds = model(bg, {'atoms':atoms_feats, 'bonds':bonds_feats})
                
            torch.cuda.empty_cache()
            eval_matrix.update(bonds_preds, bonds_labels, bonds_masks)
            
    roc_score = np.mean(eval_matrix.roc_auc_score())
    prc_score = np.mean(eval_matrix.roc_precision_recall_score()) 
    #abs_score = eval_matrix.logistic_absolute_rate()*100
    abs_score = eval_matrix.absolute_correct_rate()*100
 
    return roc_score, prc_score, abs_score 



#################################################################################################################
################################################# pyg functions #################################################
#################################################################################################################
def pyg_Hetero_train(model, data_loader, loss_func):
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


def pyg_Hetero_evaluate(model, data_loader):
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
    parser.add_argument('-g', '--graph', type=str, required=True, help='graph type (Homo or Hetero)')
    args = parser.parse_args()

    seed = 42
    setup_seed(42)
    path = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]  #D:/share/TOSS/
    
    model_path = os.path.join(path,"models")
    os.mkdir(model_path) if not os.path.exists(model_path) else None

    raw_path = os.path.join(path, "raw")
    os.mkdir(raw_path) if not os.path.exists(raw_path) else None
    shutil.copy(os.path.join(path, "length_matrix_dict.pkl"), os.path.join(raw_path, "length_matrix_dict.pkl"))

    processed_path = os.path.join(path, "processed")
    os.mkdir(processed_path) if not os.path.exists(processed_path) else None

    package_use = args.package
    graph_type = args.graph

    (atom_feats, bond_feats) = (13, 13)

    date = "%02d"%time.localtime()[1] + "%02d"%time.localtime()[2] 

    if package_use.lower() == "dgl":
        if os.path.exists(path + "/NN_LP_DGL_dataset_%s.pkl"%(graph_type)):
            print("Getting the well constructed Link Prediction dataset...")
            file_get= open(path + "/NN_LP_DGL_dataset_%s.pkl"%(graph_type),"rb")
            tr_dataset, te_dataset, vl_dataset = pickle.load(file_get)
            file_get.close()
            print("Done!")
        else:
            print("Getting the all datapoint from the TOSS result dictonary...")
            file_get= open(path + "/graphs_dict.pkl","rb")
            graphs_dict = pickle.load(file_get)
            file_get.close()
            print("Done!")

            print("Getting the length_matrix_dict...")
            file_get = open(path + "/length_matrix_dict.pkl",'rb') 
            length_matrix_dict = pickle.load(file_get) 
            file_get.close()
            for k,v in length_matrix_dict.items():
                np.fill_diagonal(v,0)
            print("Done!")

            print("Splitting the original dataset to train, validate, and test dataset...")
            tr_graphs, vl_graphs, te_graphs = split_dataset_to_three(graphs_dict,seed,ratio=[3,1,1])
            print("Done!")

            print("Constructing the three graph_dicts for Link Prediction...")
            print("Refining the original graph_dicts...")
            tr_graphs = refine_graphs_dict(tr_graphs, criterion = ["NOSELF"])
            te_graphs = refine_graphs_dict(te_graphs, criterion = ["NOSELF"])
            vl_graphs = refine_graphs_dict(vl_graphs, criterion = ["NOSELF"])
            print("Done!")

            print("Preparing the datasets...")
            if graph_type == "Hetero":
                tr_dataset = TOSS_DGL_LP_FN_DataSet(tr_graphs, length_matrix_dict)
                vl_dataset = TOSS_DGL_LP_FN_DataSet(vl_graphs, length_matrix_dict)
                te_dataset = TOSS_DGL_LP_FN_DataSet(te_graphs, length_matrix_dict)
            else: 
                assert graph_type == "Homo"
                tr_dataset = TOSS_DGL_LP_SG_DataSet(tr_graphs, length_matrix_dict)
                vl_dataset = TOSS_DGL_LP_SG_DataSet(vl_graphs, length_matrix_dict)
                te_dataset = TOSS_DGL_LP_SG_DataSet(te_graphs, length_matrix_dict)
            print("Done!")

            print("Saving the three datasets...")
            file_save= open(path + "/NN_LP_DGL_dataset_%s.pkl"%(graph_type),"wb")
            pickle.dump((tr_dataset, te_dataset, vl_dataset), file_save)
            file_save.close()
            print("Done!")
        print("All preparations are done!")

        if graph_type == "Homo":
            tr_loader = DGL_DataLoader(tr_dataset, batch_size=96, shuffle=True, collate_fn=DGL_LP_SG_collate, drop_last = True)
            vl_loader = DGL_DataLoader(vl_dataset, batch_size=96, shuffle=True, collate_fn=DGL_LP_SG_collate, drop_last = True)
            te_loader = DGL_DataLoader(te_dataset, batch_size=96, shuffle=True, collate_fn=DGL_LP_SG_collate, drop_last = True)
        else:
            assert graph_type == "Hetero"
            tr_loader = DGL_DataLoader(tr_dataset, batch_size=96, shuffle=True, collate_fn=DGL_LP_FN_collate, drop_last = True)
            vl_loader = DGL_DataLoader(vl_dataset, batch_size=96, shuffle=True, collate_fn=DGL_LP_FN_collate, drop_last = True)
            te_loader = DGL_DataLoader(te_dataset, batch_size=96, shuffle=True, collate_fn=DGL_LP_FN_collate, drop_last = True)



        dgl_Hetero_GCN_s = dgl_Hetero_GCNPredictor(atom_feats=atom_feats, bond_feats=bond_feats, hidden_feats=[256,256,256,256],Etypes=["interacts"],
                                                   gnn_norm=None, activation=None, residual=None, batchnorm=None, dropout=None,
                                                   predictor_hidden_feats=64, n_tasks=2,predictor_dropout=0.3)


        dgl_Hetero_GCN_w = dgl_Hetero_GCNPredictor(atom_feats=atom_feats, bond_feats=bond_feats, hidden_feats=[64,64],Etypes=["interacts"],
                                                   gnn_norm=None, activation=None, residual=None, batchnorm=None, dropout=None,
                                                   predictor_hidden_feats=64, n_tasks=2,predictor_dropout=0.3)


    if package_use == "pyg":
        print("Getting the processed datasets...")
        dataset = TOSS_PYG_LP_FN_DataSet(root=path)
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


        pyg_Hetero_GCN_s = pyg_Hetero_GCNPredictor(atom_feats=atom_feats, bond_feats=bond_feats, hidden_feats=[256,256,256,256], 
                                                 activation=None, residual=None, batchnorm=None, dropout=None,
                                                 predictor_hidden_feats=64, n_tasks=2,predictor_dropout=0.3)

        pyg_Hetero_GCN_w = pyg_Hetero_GCNPredictor(atom_feats=atom_feats, bond_feats=bond_feats, hidden_feats=[64,64], 
                                                 activation=None, residual=None, batchnorm=None, dropout=None,
                                                 predictor_hidden_feats=64, n_tasks=2,predictor_dropout=0.3)


    exec("train=%s_%s_train"%(package_use, graph_type))
    exec("evaluate=%s_%s_evaluate"%(package_use, graph_type))

    model_name = args.package.lower() + "_" + args.graph + "_" + args.model.upper() + "_" + args.size.lower()
    exec("model = %s"%model_name)


    optimizer = torch.optim.Adam(model.parameters(), lr = 10 ** -3, weight_decay = 10 ** -4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4, eps=1e-8)
    loss_func = BCEWithLogitsLoss() #should get the labels data!!!!

    f_name = os.path.join(model_path, "%s_%s.txt"%(model_name, date))
    m_name = os.path.join(model_path, "%s_%s.pth"%(model_name, date))

    print("Model Trainning...")
    parameter_space = sum(list(map(lambda x:model.state_dict()[x].reshape(-1,1).shape[0],list(model.state_dict().keys()))))
    print("The total number of parameters of the Model is %s"%parameter_space)
    epochs = 300
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
        scheduler.step(loss)

        print("Epoch: %s, Cost: %s, tr_roc:%s, tr_prc:%s, tr_abs:%s, vl_roc:%s, vl_prc:%s, vl_abs:%s, te_roc:%s, te_prc:%s, te_abs:%s."
            %(epoch, cost, tr_roc, tr_prc, tr_abs, vl_roc, vl_prc, vl_abs, te_roc, te_prc, te_abs))

        with open(f_name,mode="a",encoding="utf-8") as f:
            data=f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"
                %(epoch, cost, tr_roc, tr_prc, tr_abs, vl_roc, vl_prc, vl_abs, te_roc, te_prc, te_abs))

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
"""END HERE"""


