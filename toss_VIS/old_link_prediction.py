import pickle
import time
import random
from model_utils_dgl import *
from dataset_utils_dgl import *
from data_utils import *
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from torch.nn import BCEWithLogitsLoss,BCELoss


work_type = ["PRE", "PST"][0]
graph_type = ["Hetero", "Homo"][0]


#################################################################################################################
################################################PRESET PARAMETERS################################################
#################################################################################################################
def train(model, data_loader, loss_func):
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


def evaluate(model, data_loader):
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


def Hetero_train(model, data_loader, loss_func):
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


def Hetero_evaluate(model, data_loader):
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
################################################################################################################
###############################################SAME FOR ALL MODEL###############################################
################################################################################################################
seed = 42
path = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]

model_path = path + "/models"

if not os.path.exists(model_path):
    os.mkdir(model_path)

try:
    print("Getting the well constructed Link Prediction dataset...")
    file_get= open(path + "/NN_LP_split_dataset_%s_%s.pkl"%(graph_type, work_type),"rb")
    tr_dataset, te_dataset, vl_dataset = pickle.load(file_get)
    file_get.close()
    print("Done!")

except:
    print("Failed to get the well constructed Link Prediction dataset!!!")
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

    if work_type == "PRE":
        print("DO the Link Prediction before TOSS!")
        None_list = list(map(lambda x:tr_graphs[x]["n"].drop(["CN","SEN","OS"], axis=1, inplace=True), list(tr_graphs.keys())))
        None_list = list(map(lambda x:te_graphs[x]["n"].drop(["CN","SEN","OS"], axis=1, inplace=True), list(te_graphs.keys())))
        None_list = list(map(lambda x:vl_graphs[x]["n"].drop(["CN","SEN","OS"], axis=1, inplace=True), list(vl_graphs.keys())))
        print("Done!")

    print("Preparing the datasets...")
    if graph_type == "Hetero":
        tr_dataset = TOSS_LP_FN_DataSet(tr_graphs, length_matrix_dict)
        vl_dataset = TOSS_LP_FN_DataSet(vl_graphs, length_matrix_dict)
        te_dataset = TOSS_LP_FN_DataSet(te_graphs, length_matrix_dict)
    else: 
        assert graph_type == "Homo"
        tr_dataset = TOSS_LP_SG_DataSet(tr_graphs, length_matrix_dict)
        vl_dataset = TOSS_LP_SG_DataSet(vl_graphs, length_matrix_dict)
        te_dataset = TOSS_LP_SG_DataSet(te_graphs, length_matrix_dict)
    print("Done!")

    print("Saving the three datasets...")
    file_save= open(path + "/NN_LP_split_dataset_%s_%s.pkl"%(graph_type, work_type),"wb")
    pickle.dump((tr_dataset, te_dataset, vl_dataset), file_save)
    file_save.close()
    print("Done!")

    print("All preparations are done!")
################################################################################################################
#############################################HYPERPARAMETERS####################################################
################################################################################################################

if __name__ == "__main__":

    seed=42
    setup_seed(seed)
    (atom_feats, bond_feats) = (13,13) if work_type == "PRE" else (16,13)

    """
    GCN_w = my_GCNLinkPredictor(in_feats=input_size, hidden_feats=[64, 64], 
                                gnn_norm=None, activation=None, residual=None, batchnorm=None, dropout=None,
                                predictor="MLP")

    GCN_s = my_GCNLinkPredictor(in_feats=input_size, hidden_feats=[128, 128], 
                                gnn_norm=None, activation=None, residual=None, batchnorm=None, dropout=None,
                                predictor="MLP")

    GAT_w = my_GATLinkPredictor(in_feats=input_size, hidden_feats=[64, 64], num_heads=[4, 4], 
                                feat_drops=None, attn_drops=None, alphas=None, residuals=None, agg_modes=None, 
                                activations=None, biases=None, 
                                predictor="MLP") 

    GAT_s = my_GATLinkPredictor(in_feats=input_size, hidden_feats=[256, 256, 256, 256], num_heads=[4, 4, 4, 4], 
                                feat_drops=None, attn_drops=None, alphas=None, residuals=None, agg_modes=None, 
                                activations=None, biases=None, 
                                predictor="MLP")   

    AFP_w = my_AFPLinkPredictor(node_feat_size=input_size, edge_feat_size=1, dropout=0., 
                                num_layers=2, graph_feat_size=64,
                                predictor="MLP")

    AFP_s = my_AFPLinkPredictor(node_feat_size=input_size, edge_feat_size=9, dropout=0., 
                                num_layers=6, graph_feat_size=256,
                                predictor="MLP")

    MPNN_w = my_MPNNLinkPredictor(node_in_feats=input_size, edge_in_feats=1, 
                                  num_step_message_passing=2, node_out_feats=64, edge_hidden_feats=64,
                                  predictor="MLP") 

    MPNN_s = my_MPNNLinkPredictor(node_in_feats=input_size, edge_in_feats=9, 
                                  num_step_message_passing=4, node_out_feats=128, edge_hidden_feats=128,
                                  predictor="MLP")
    """

    Hetero_GCN_s = dgl_Hetero_GCNPredictor(atom_feats=atom_feats, bond_feats=bond_feats, hidden_feats=[256,256,256,256],Etypes=["interacts"],
                                          gnn_norm=None, activation=None, residual=None, batchnorm=None, dropout=None,
                                          predictor_hidden_feats=64, n_tasks=2,predictor_dropout=0.3)


    Hetero_GCN_w = dgl_Hetero_GCNPredictor(atom_feats=atom_feats, bond_feats=bond_feats, hidden_feats=[64,64],Etypes=["interacts"],
                                          gnn_norm=None, activation=None, residual=None, batchnorm=None, dropout=None,
                                          predictor_hidden_feats=64, n_tasks=2,predictor_dropout=0.3)

    #model_name = str(input("Give the name of the model from 'GCN_w, GCN_s, GAT_w, GAT_s, AFP_w, AFP_s, MPNN_w, MPNN_s':"))
    model_name = "Hetero_GCN_s"
    date = "%02d"%time.localtime()[1] + "%02d"%time.localtime()[2] 

    if graph_type == "Homo":
        tr_loader = DataLoader(tr_dataset, batch_size=96, shuffle=True, collate_fn=LP_SG_collate, drop_last = True)
        vl_loader = DataLoader(vl_dataset, batch_size=96, shuffle=True, collate_fn=LP_SG_collate, drop_last = True)
        te_loader = DataLoader(te_dataset, batch_size=96, shuffle=True, collate_fn=LP_SG_collate, drop_last = True)
    else:
        assert graph_type == "Hetero"
        tr_loader = DataLoader(tr_dataset, batch_size=96, shuffle=True, collate_fn=LP_FN_collate, drop_last = True)
        vl_loader = DataLoader(vl_dataset, batch_size=96, shuffle=True, collate_fn=LP_FN_collate, drop_last = True)
        te_loader = DataLoader(te_dataset, batch_size=96, shuffle=True, collate_fn=LP_FN_collate, drop_last = True)


    exec("model = %s"%model_name)
    parameter_space = sum(list(map(lambda x:model.state_dict()[x].reshape(-1,1).shape[0],list(model.state_dict().keys()))))
    print(parameter_space)

    optimizer = torch.optim.Adam(model.parameters(), lr = 10 ** -3, weight_decay = 10 ** -4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4, eps=1e-8)

    #loss_func = compute_loss
    #loss_func = BCEWithLogitsLoss(reduction="none") #should get the labels data!!!!
    loss_func = BCEWithLogitsLoss() #should get the labels data!!!!

    f_name = "%s_%s_%s.txt"%(model_name, work_type, date)

    print("Model is training...")
    epochs = 300
    best, patience = 0,0
    for epoch in range(epochs):
        s_time = time.time()

        tr_roc, tr_prc, tr_abs, loss = Hetero_train(model, tr_loader, loss_func)
        tr_roc, tr_prc, tr_abs = "%.6f"%tr_roc, "%.6f"%tr_prc, "%.2f"%tr_abs

        vl_roc, vl_prc, vl_abs = Hetero_evaluate(model, vl_loader)
        vl_roc, vl_prc, vl_abs = "%.6f"%vl_roc, "%.6f"%vl_prc, "%.2f"%vl_abs

        te_roc, te_prc, te_abs = Hetero_evaluate(model, te_loader)
        te_roc, te_prc, te_abs = "%.6f"%te_roc, "%.6f"%te_prc, "%.2f"%te_abs

        e_time = time.time()
        cost = e_time-s_time
        cost = "%.2f"%cost
        scheduler.step(loss)

        print("Epoch: %s, Cost: %s, tr_roc:%s, tr_prc:%s, tr_abs:%s, vl_roc:%s, vl_prc:%s, vl_abs:%s, te_roc:%s, te_prc:%s, te_abs:%s."
            %(epoch, cost, tr_roc, tr_prc, tr_abs, vl_roc, vl_prc, vl_abs, te_roc, te_prc, te_abs))

        with open(file=model_path + "/%s"%f_name,mode="a",encoding="utf-8") as f:
            data=f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"
                %(epoch, cost, tr_roc, tr_prc, tr_abs, vl_roc, vl_prc, vl_abs, te_roc, te_prc, te_abs))

        learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        best, patience, stop_flag, save_flag = stopper(vl_abs, best, epoch, patience, lr=learning_rate, max_patience=20, tolerance=0.01)

        if save_flag:
            torch.save(model.state_dict(), model_path + "/%s.pth"%f_name[0:-4])
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