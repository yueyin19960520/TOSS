import torch
from torch import nn
import torch.nn.functional as F
#from torch_geometric.nn import GraphConv
# from torch_geometric.nn import GATConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import reset, glorot, zeros
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.nn.module_dict import ModuleDict
from collections import defaultdict
from torch_geometric.nn.conv.hgt_conv import group


#########################################################################################################################
################################################## NODE CLASSIFICATION ##################################################
#########################################################################################################################
class pyg_MLPNodeClassificationPredictor(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, n_tasks, dropout=0.):
        super(pyg_MLPNodeClassificationPredictor, self).__init__()

        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_feats),
            nn.Linear(hidden_feats, n_tasks)
        )

    def forward(self, feats):
        return self.predict(feats)


#################################################
# GCN(GCNLayer) + MLPPredictor --> GCNPredictor #
#################################################
class pyg_GraphConv(MessagePassing):

    def __init__(self, in_channels, out_channels, aggr='add', bias=True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_rel = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_root = nn.Linear(in_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_rel.reset_parameters()
        #self.lin_root.reset_parameters()

    """
    def forward(self, x, edge_index):  #make it same as the DGL package algorithm
        x = self.lin_rel(x)
        out = self.propagate(edge_index, x=x)
        return out

    """
    def forward(self, x, edge_index):   # here is the original script from the pyg package, but more parameters!!!!

        x = (x, x)
        out = self.propagate(edge_index, x=x)
        out = self.lin_rel(out)

        x_r = x[1]
        if x_r is not None:
            out = out + self.lin_root(x_r)
        return out
    

class pyg_GCNLayer(torch.nn.Module):
    def __init__(self, in_feats, out_feats, activation,    
                 residual, batchnorm, dropout):
        super(pyg_GCNLayer, self).__init__()

        self.activation = activation
        
        self.graph_conv = pyg_GraphConv(in_channels=in_feats, out_channels=out_feats, aggr="add")
        # which calculates the new feats without any normalization here, just add all embeddings of all neighbors
        # X_i^prime = W_root * X_i^0 + W_rel * (\sum_(i's Neighbors j) e_ij * X_j^0) + b_rel
        
        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats, bias = True)
            
        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)
            # when calculate the mean, the momentum works, which means that E[x]'=momentum*E[x] 
            # so when it does the calculation, the Var[x] is using biased variable (unbiased=False) 
            # but when storing the running_var, the result is the unbiased varivable (unbiased=True)
            
        self.dropout = nn.Dropout(dropout)
        
    def reset_parameters(self):
        self.graph_conv.reset_parameters()

        if self.residual:
            self.res_connection.reset_parameters()

        if self.bn:
            self.bn_layer.reset_parameters()

    def forward(self, feats, edge_index):
        new_feats = self.activation(self.graph_conv(feats, edge_index)) #lacking of the g

        if self.residual:
            res_feats = self.activation(self.res_connection(feats))
            new_feats = new_feats + res_feats

        new_feats = self.dropout(new_feats)

        if self.bn:
            new_feats = self.bn_layer(new_feats)

        return new_feats


class pyg_GCN(torch.nn.Module):
    def __init__(self, in_feats=None, hidden_feats=None, activation=None, 
                 residual=None, batchnorm=None, dropout=None):
        super(pyg_GCN, self).__init__()

        n_layers = len(hidden_feats)

        if activation is None:
            activation = [F.relu for _ in range(n_layers)]
        if residual is None:
            residual = [True for _ in range(n_layers)]
        if batchnorm is None:
            batchnorm = [True for _ in range(n_layers)]
        if dropout is None:
            dropout = [0. for _ in range(n_layers)]
        
        self.gnn_layers = torch.nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(pyg_GCNLayer(in_feats, hidden_feats[i], activation[i], 
                                               residual[i], batchnorm[i], dropout[i]))
            in_feats = hidden_feats[i]

    def reset_parameters(self):
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, feats, edge_index):
        for gnn in self.gnn_layers:
            feats = gnn(feats, edge_index)
        return feats


class pyg_GCNPredictor(torch.nn.Module):
    
    def __init__(self, in_feats, hidden_feats=None, 
                 activation=None, residual=None, batchnorm=None, dropout=None, 
                 predictor_hidden_feats=None, n_tasks=None, predictor_dropout=None):
        super(pyg_GCNPredictor, self).__init__()

        self.gnn = pyg_GCN(in_feats=in_feats, hidden_feats=hidden_feats, activation=activation, 
                          residual=residual, batchnorm=batchnorm, dropout=dropout)
        
        gnn_out_feats = hidden_feats[-1]

        self.predict = pyg_MLPNodeClassificationPredictor(gnn_out_feats, predictor_hidden_feats, n_tasks, predictor_dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        node_feats = self.gnn(x, edge_index)
        predicted_feats = self.predict(node_feats)
        return predicted_feats


#################################################
# GAT(GATLayer) + MLPPredictor --> GCNPredictor #
#################################################
class pyg_GATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads,negative_slope=0.2, dropout=0.0,bias=True,
                 concat=True, add_self_loops=True,      
                 edge_dim=None, fill_value='mean', **kwargs,):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.heads = heads
        self.out_channels = out_channels
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        self.lin_src = nn.Linear(in_channels, heads * out_channels, bias=False) # weight_initializer='glorot'
        #self.lin_dst = self.lin_src

        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.residual = True
        if self.residual:
            self.res_fc = nn.Linear(in_channels, heads * out_channels, bias=False) 

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_src.reset_parameters()
        self.res_fc.reset_parameters()
        #self.lin_dst.reset_parameters()
        reset(self.att_src)
        reset(self.att_dst)
        zeros(self.bias)

    def forward(self, x, edge_index, return_attention_weights=None):

        H, C = self.heads, self.out_channels
        X = x
        assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
        x_src = x_dst = self.lin_src(x).view(-1, H, C)
        x = (x_src, x_dst)

        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = (x_dst * self.att_dst).sum(dim=-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            num_nodes = x_src.size(0)
            edge_index, edge_attr = remove_self_loops(edge_index)
            edge_index, edge_attr = add_self_loops(edge_index, num_nodes=num_nodes)

        alpha = self.edge_updater(edge_index, alpha=alpha)

        out = self.propagate(edge_index, x=x, alpha=alpha)

        #if self.concat:
            #out = out.view(-1, self.heads * self.out_channels)
        #else:
            #out = out.mean(dim=1)

        if self.residual:
            res = self.res_fc(X).view(-1, H, C)
            out = res + out

        if self.bias is not None:
            out = out + self.bias.view(-1, H, C)

        return out

    def edge_update(self, alpha_j, alpha_i, index):

        alpha =  alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index)
        alpha = F.dropout(alpha, p=self.dropout)
        return alpha

    def message(self, x_j, alpha):
        return alpha.unsqueeze(-1) * x_j # make the shape of the alpha from [xxx, 1] to [xxx]


class pyg_GATLayer(nn.Module):

    def __init__(self, in_feats, out_feats, num_heads, dropout, alpha, bias, activation, agg_modes): 

        super(pyg_GATLayer, self).__init__()

        self.activation = activation
        self.agg_modes = agg_modes

        self.gat_conv = pyg_GATConv(in_channels=in_feats, out_channels=out_feats, heads=num_heads, 
                                dropout=dropout, bias=bias, negative_slope=alpha, add_self_loops=True)
        
        self.bn = True
        if self.bn:
            if agg_modes == "flatten":
                self.bn_layer = nn.BatchNorm1d(out_feats * num_heads)
            else:
                self.bn_layer = nn.BatchNorm1d(out_feats)
 
    def reset_parameters(self):
        self.gat_conv.reset_parameters()

        if self.bn:
            self.bn_layer_flatten.reset_parameters()

    def forward(self, feats, edge_index):

        new_feats = self.gat_conv(feats, edge_index)

        if self.agg_modes == "flatten":
            new_feats = new_feats.flatten(1)
        else:
            new_feats = new_feats.mean(1)

        if self.activation is not None:
            new_feats = self.activation(new_feats)

        if self.bn:
            new_feats = self.bn_layer(new_feats)

        return new_feats


class pyg_GAT(nn.Module):

    def __init__(self, in_feats, hidden_feats=None, num_heads=None, dropouts=None, alphas=None, biases=None, activations=None, agg_modes=None):
        super(pyg_GAT, self).__init__()

        n_layers = len(hidden_feats)
        
        # Set default values if not provided
        if activations is None:
            activations = [F.elu for _ in range(n_layers - 1)]
            activations.append(None)
        if num_heads is None:
            num_heads = [4 for _ in range(n_layers)]
        if dropouts is None:
            dropouts = [0. for _ in range(n_layers)]
        if alphas is None:
            alphas = [0.2 for _ in range(n_layers)]
        if biases is None:
            biases = [True for _ in range(n_layers)]
        if agg_modes is None:
            agg_modes = ["flatten" for _ in range(n_layers - 1)]
            agg_modes.append("mean")

        # The main difference here is that in PyG, we don't need to pass the graph to the forward function,
        # instead, we pass the edge indices.
        self.hidden_feats = hidden_feats
        self.num_heads = num_heads
        self.agg_modes = agg_modes
        self.gnn_layers = nn.ModuleList()

        for i in range(n_layers):
            self.gnn_layers.append(pyg_GATLayer(in_feats, hidden_feats[i], num_heads[i], dropouts[i], 
                                                biases[i], alphas[i], activations[i], agg_modes[i]))
            in_feats = hidden_feats[i] * num_heads[i]

    def reset_parameters(self):
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, feats, edge_index):
        for gnn in self.gnn_layers:
            feats = gnn(feats, edge_index)
        return feats

class pyg_GATPredictor(nn.Module):
    def __init__(self, in_feats, hidden_feats=None, num_heads=None, dropouts=None, biases=None, alphas=None, 
                 agg_modes=None, activations=None,
                 predictor_hidden_feats=None, n_tasks=None, predictor_dropout=None):
        super(pyg_GATPredictor, self).__init__()

        self.gnn = pyg_GAT(in_feats=in_feats, hidden_feats=hidden_feats, num_heads=num_heads, 
                           dropouts=dropouts, biases=biases, alphas=alphas, 
                           agg_modes=agg_modes, activations=activations)

        if self.gnn.agg_modes[-1] == "flatten":
            #gnn_out_feats = self.gnn.gnn_layers[-1].gat_conv.out_channels * self.gnn.gnn_layers[-1].gat_conv.heads
            gnn_out_feats = self.gnn.hidden_feats[-1] * self.gnn.num_heads[-1]
        else:
            gnn_out_feats = self.gnn.hidden_feats[-1]

        self.predict = pyg_MLPNodeClassificationPredictor(gnn_out_feats, predictor_hidden_feats, n_tasks, predictor_dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        node_feats = self.gnn(x, edge_index)
        predicted_feats = self.predict(node_feats)
        return predicted_feats


#################################################
# AFP(AFPLayer) + MLPPredictor --> AFPPredictor #
#################################################
class pyg_AttentiveGRU1(MessagePassing):
    def __init__(self, node_feat_size, edge_feat_size, edge_hidden_size, dropout):
        super(pyg_AttentiveGRU1, self).__init__(aggr='add', flow='source_to_target')

        self.edge_transform = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(edge_feat_size, edge_hidden_size))
        
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def reset_parameters(self):
        self.edge_transform[1].reset_parameters()
        self.gru.reset_parameters()

    def forward(self, edge_index, edge_logits, edge_feats, node_feats):
        edge_feats_transformed = self.edge_transform(edge_feats)
        edge_weights = softmax(edge_logits, edge_index[0])
        edge_weights = edge_weights.view(-1, 1) # share as [XXX, 1] but logits shape as [XXX]
        aggr_out = self.propagate(edge_index, edge_weights=edge_weights, edge_feats=edge_feats_transformed, node_feats=node_feats)
        return aggr_out

    def message(self, edge_weights, edge_feats):
        # the aggr mode has been defined in the init as "add" and from "source_to_target"
        return edge_weights * edge_feats 

    def update(self, aggr_out, node_feats):
        # the aggr_out is the output from the message, which is edge_weights * edge_feats here
        # the node_feats is any useful kargs
        context = F.elu(aggr_out)
        return F.relu(self.gru(context, node_feats))


class pyg_AttentiveGRU2(MessagePassing):
    def __init__(self, node_feat_size, edge_hidden_size, dropout):
        super(pyg_AttentiveGRU2, self).__init__(aggr='add',flow='source_to_target')

        self.project_node = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(node_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def reset_parameters(self):
        self.project_node[1].reset_parameters()
        self.gru.reset_parameters()

    def forward(self, edge_index, edge_logits, node_feats):
        node_feats_proj = self.project_node(node_feats)
        edge_weights = softmax(edge_logits, edge_index[0])
        edge_weights = edge_weights.view(-1, 1)
        return self.propagate(edge_index, edge_weights=edge_weights, node_feats=node_feats_proj) # aggr_mode = add, cool!

    def message(self, edge_weights, node_feats_j):
        return edge_weights * node_feats_j

    def update(self, aggr_out, node_feats):
        context = F.elu(aggr_out)
        return F.relu(self.gru(context, node_feats))


class pyg_GetContext(MessagePassing):
    def __init__(self, node_feat_size, edge_feat_size, graph_feat_size, dropout):
        super(pyg_GetContext, self).__init__(aggr='add')

        self.project_node = nn.Sequential(nn.Linear(node_feat_size, graph_feat_size), nn.LeakyReLU())
        self.project_edge1 = nn.Sequential(nn.Linear(node_feat_size + edge_feat_size, graph_feat_size), nn.LeakyReLU())
        self.project_edge2 = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * graph_feat_size, 1), nn.LeakyReLU())
        self.attentive_gru = pyg_AttentiveGRU1(graph_feat_size, graph_feat_size, graph_feat_size, dropout)

    def reset_parameters(self):
        self.project_node[0].reset_parameters()
        self.project_edge1[0].reset_parameters()
        self.project_edge2[1].reset_parameters()
        self.attentive_gru.reset_parameters()

    def forward(self, edge_index, node_feats, edge_feats):
        hv = node_feats
        hv_new = self.project_node(hv)
        he = edge_feats
        edge_feats_proj1 = self.project_edge1(torch.cat([hv[edge_index[0]], he], dim=1))
        # concat the src_feats and the edge_feats
        edge_feats_proj2 = self.project_edge2(torch.cat([hv_new[edge_index[1]], edge_feats_proj1], dim=1))
        # concat the dst_feats and the edge_feats_project_1 aka the new projected edge_feats

        edge_logits = edge_feats_proj2.squeeze(-1)
        return self.attentive_gru(edge_index, edge_logits, edge_feats_proj1, hv_new)


class pyg_GNNLayer(MessagePassing): 
    def __init__(self, node_feat_size, graph_feat_size, dropout):
        super(pyg_GNNLayer, self).__init__(aggr='add')

        self.project_edge = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * node_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = pyg_AttentiveGRU2(node_feat_size, graph_feat_size, dropout)

    def reset_parameters(self):
        self.project_edge[1].reset_parameters()
        self.attentive_gru.reset_parameters()

    def forward(self, edge_index, node_feats):
        edge_feats_proj = self.project_edge(torch.cat([node_feats[edge_index[0]], node_feats[edge_index[1]]], dim=1))
        edge_logits = edge_feats_proj.squeeze(-1)  # may cause some issue!!!!!!!!!!!!
        return self.attentive_gru(edge_index, edge_logits, node_feats)


class pyg_AFPPredictor(torch.nn.Module):
    def __init__(self, node_feat_size=None, edge_feat_size=None, graph_feat_size=None, dropout=None,
                 num_layers=None,
                 predictor_hidden_feats=None, n_tasks=None, predictor_dropout=None):
        super(pyg_AFPPredictor, self).__init__()

        self.init_context = pyg_GetContext(node_feat_size, edge_feat_size, graph_feat_size, dropout)
        
        self.gnn_layers = nn.ModuleList()

        for _ in range(num_layers - 1):
            self.gnn_layers.append(pyg_GNNLayer(graph_feat_size, graph_feat_size, dropout))
        
        self.predict = pyg_MLPNodeClassificationPredictor(graph_feat_size, predictor_hidden_feats, n_tasks, predictor_dropout)

    def reset_parameters(self):
        self.init_context.reset_parameters()
        
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, data):
        edge_index, node_feats, edge_feats = data.edge_index, data.x, data.edge_attr

        edge_index, _ = add_self_loops(edge_index, num_nodes=node_feats.size(0))
        loop_edge_feats = torch.zeros(node_feats.size(0), edge_feats.size(1)).to(edge_feats.device)
        edge_feats = torch.cat([edge_feats, loop_edge_feats], dim=0)

        node_feats = self.init_context(edge_index, node_feats, edge_feats)

        for gnn in self.gnn_layers:
            node_feats = gnn(edge_index, node_feats)

        predicted_feats = self.predict(node_feats)
        return predicted_feats


####################################################
# MPNN(MPNNLayer) + MLPPredictor --> MPNNPredictor #
####################################################
class PyG_NNConv(MessagePassing):
    def __init__(self, in_feats, out_feats, edge_func):
        super(PyG_NNConv, self).__init__(aggr='mean')  # "Mean" aggregation.
        self.lin_node = nn.Linear(in_feats, out_feats)
        self.edge_func = edge_func

    def forward(self,edge_index, x, edge_attr):
        
        x = self.lin_node(x)
        edge_attr = self.edge_func(edge_attr).view(-1, self.lin_node.out_features, self.lin_node.out_features)

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return torch.bmm(x_j.unsqueeze(1), edge_attr).squeeze(1)


class PyG_MPNNGNN(torch.nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, node_out_feats, edge_hidden_feats, num_step_message_passing):
        super(PyG_MPNNGNN, self).__init__()

        self.project_node_feats = nn.Sequential(nn.Linear(node_in_feats, node_out_feats), 
                                                nn.ReLU())
        
        self.num_step_message_passing = num_step_message_passing

        edge_network = nn.Sequential(nn.Linear(edge_in_feats, edge_hidden_feats),
                                     nn.ReLU(),
                                     nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats))
        
        self.conv = PyG_NNConv(in_feats=node_out_feats, out_feats=node_out_feats, edge_func=edge_network, )

        self.gru = nn.GRU(node_out_feats, node_out_feats)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        node_feats = self.project_node_feats(x)
        hidden_feats = node_feats.unsqueeze(0)

        for i in range(self.num_step_message_passing):
            node_feats = F.relu(self.conv(edge_index, node_feats, edge_attr))
            node_feats, _ = self.gru(node_feats.unsqueeze(0), hidden_feats)
            node_feats = node_feats.squeeze(0)

        return node_feats


class PyG_MPNNPredictor(torch.nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, node_out_feats, edge_hidden_feats, num_step_message_passing,
                 n_tasks, predictor_hidden_feats, predictor_dropout):
        
        super(PyG_MPNNPredictor, self).__init__()

        self.gnn = PyG_MPNNGNN(node_in_feats, edge_in_feats, node_out_feats, edge_hidden_feats,
                               num_step_message_passing)

        self.predict = pyg_MLPNodeClassificationPredictor(node_out_feats, predictor_hidden_feats, n_tasks, predictor_dropout)

    def forward(self, data):
        node_feats = self.gnn(data)
        predicted_feats = self.predict(node_feats)
        return predicted_feats


#########################################################################################################################
################################################ BOND VALENCE REGRESSION ################################################
#########################################################################################################################
class pyg_MLPBondValencePredictor(MessagePassing):
    def __init__(self, in_feats, hidden_feats, dropout):
        super(pyg_MLPBondValencePredictor, self).__init__(aggr='add',flow='source_to_target')
        
        self.layers = nn.Sequential(
            nn.Linear(2 * in_feats, hidden_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_feats, 1))
    
    def forward(self, node_feats, edge_index, edge_weights): 
        return self.propagate(edge_index, size=(node_feats.size(0), node_feats.size(0)), x=node_feats, edge_weights=edge_weights)
    
    def message(self, x_i, x_j, edge_weights):
        edge_feats = torch.cat([x_i, x_j], dim=-1)
        edge_preds = self.layers(edge_feats) * edge_weights.unsqueeze(-1)
        return edge_preds

#################################################
#GCN(GCNLayer) + BVPredictor --> GCNBVPredictor #
#################################################
class pyg_GCNBondValencePredictor(torch.nn.Module):

    def __init__(self, in_feats, hidden_feats=None, 
                 activation=None, residual=None, batchnorm=None, dropout=None, 
                 predictor_hidden_feats=None, predictor_dropout=None):
        super(pyg_GCNBondValencePredictor, self).__init__()

        self.gnn = pyg_GCN(in_feats=in_feats, hidden_feats=hidden_feats, activation=activation, 
                          residual=residual, batchnorm=batchnorm, dropout=dropout)
        
        gnn_out_feats = hidden_feats[-1]

        self.edge_predictor = pyg_MLPBondValencePredictor(gnn_out_feats, predictor_hidden_feats, predictor_dropout)

    def forward(self, data):
        x, edge_index, edge_weights, batch = data.x, data.edge_index, data.edge_weights, data.batch
        node_feats = self.gnn(x, edge_index)

        edge_scalars = self.edge_predictor(node_feats, edge_index, edge_weights)

        return edge_scalars.squeeze(-1)



#########################################################################################################################
#################################################### LINK PREDICTION ####################################################
#########################################################################################################################
class pyg_MLPLinkPredictionPredictor(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, n_tasks, dropout=0.):
        super(pyg_MLPLinkPredictionPredictor, self).__init__()

        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_feats),
            nn.Linear(hidden_feats, n_tasks)
        )

    def forward(self, feats):
        return self.predict(feats)


##################################################
# GCN(GCNLayer) + BVPredictor --> GCNBVPredictor #
##################################################
class pyg_Hetero_Conv(torch.nn.Module):
    def __init__(self, convs, aggr="sum"):
        super().__init__()

        #self.convs = ModuleDict({'__'.join(k): v for k, v in convs.items()})
        self.convs = convs
        self.aggr = aggr

    def reset_parameters(self):
        for conv in self.convs.values():
            conv.reset_parameters()

    def forward(self,x_dict, edge_index_dict):
        out_dict = defaultdict(list)
        for edge_type, edge_index in edge_index_dict.items():
            
            src, rel, dst = edge_type
            str_edge_type = '__'.join(edge_type)
            #conv = self.convs[str_edge_type]
            conv = self.convs

            #if src == dst:
                #out = conv(x_dict[src], edge_index)
            #else:
            out = conv((x_dict[src], x_dict[dst]), edge_index)

            out_dict[dst].append(out)

        for key, value in out_dict.items():
            out_dict[key] = group(value, self.aggr)

        return out_dict

class pyg_Hetero_GraphConv(MessagePassing):

    def __init__(self, in_feats, out_feats, aggr='add', bias=True, **kwargs):
        super().__init__(aggr=aggr, flow='source_to_target', **kwargs)

        self.in_feats = in_feats
        self.out_feats = out_feats

        self.lin_rel = nn.Linear(in_feats, out_feats, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin_rel.weight)
        torch.nn.init.zeros_(self.lin_rel.bias)

    def forward(self, x, edge_index):  #make it same as the DGL package algorithm
        x_src, x_dst = x
        x_src = self.lin_rel(x_src)  
        x_dst = self.lin_rel(x_dst)  
        x = (x_src, x_dst)
        out = self.propagate(edge_index, x=x)
        #out = torch.mm(out, torch.ones((self.in_feats, self.out_feats))) #self.lin_rel(out)
        return out

    def message_and_aggregate(self, adj_t, x):
        return spmm(adj_t, x[0], reduce=self.aggr)
    
    
class pyg_Hetero_GCNLayer(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, activation, residual, batchnorm, dropout):
        super(pyg_Hetero_GCNLayer, self).__init__()

        self.activation = activation
        """self.graph_conv = pyg_HeteroConv({("atoms", "interacts", "bonds"):pyg_HeteroGraphConv(in_feats, hidden_feats),
                                          ("bonds", "interacts", "atoms"):pyg_HeteroGraphConv(in_feats, hidden_feats)}, aggr="sum")"""
        self.graph_conv = pyg_Hetero_Conv(pyg_Hetero_GraphConv(in_feats, hidden_feats), aggr="sum")
        
        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, hidden_feats, bias = True)
            
        self.bn = batchnorm
        if batchnorm:
            self.bn_layer_1 = nn.BatchNorm1d(hidden_feats)
            self.bn_layer_2 = nn.BatchNorm1d(hidden_feats)
            self.bn_dict = {"atoms":self.bn_layer_1, "bonds":self.bn_layer_2}

        self.dropout = nn.Dropout(dropout)
        
    def reset_parameters(self):
        self.graph_conv.reset_parameters()

        if self.residual:
            self.res_connection.reset_parameters()

        if self.bn:
            self.bn_layer_1.reset_parameters()
            self.bn_layer_2.reset_parameters()

    def forward(self, x_dict, edge_index_dict):
        new_feats = self.graph_conv(x_dict, edge_index_dict)

        if self.residual:
            res_feats = {k: self.activation(self.res_connection(v)) for k,v in x_dict.items()}
            new_feats = {k: v + res_feats[k] for k,v in new_feats.items()}

        new_feats = {k: self.dropout(v) for k,v in new_feats.items()}

        if self.bn:
            #new_feats = {k: f(v) for (k,v),f in zip(new_feats.items(), [self.bn_layer_1, self.bn_layer_2])}
            new_feats = {k: self.bn_dict[k](v) for k,v in new_feats.items()}

        return new_feats

    
class pyg_Hetero_GCN(nn.Module):
    def __init__(self, in_feats=None, hidden_feats=None, activation=None, residual=None, batchnorm=None, dropout=None):
        super(pyg_Hetero_GCN, self).__init__()

        n_layers = len(hidden_feats)

        if activation is None:
            activation = [F.relu for _ in range(n_layers)]
        if residual is None:
            residual = [True for _ in range(n_layers)]
        if batchnorm is None:
            batchnorm = [True for _ in range(n_layers)]
        if dropout is None:
            dropout = [0. for _ in range(n_layers)]
        
        self.gnn_Hetero_layers = torch.nn.ModuleList()
        for i in range(n_layers):
            self.gnn_Hetero_layers.append(pyg_Hetero_GCNLayer(in_feats, hidden_feats[i], 
                                                              activation[i], residual[i], batchnorm[i], dropout[i]))
            in_feats = hidden_feats[i]

    def reset_parameters(self):
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, x_dict, edge_index_dict):
        for gnn in self.gnn_Hetero_layers:
            x_dict = gnn(x_dict, edge_index_dict)
        return x_dict
    
    
class pyg_Hetero_GCNPredictor(nn.Module):
    
    def __init__(self, atom_feats, bond_feats, hidden_feats=None, 
                 activation=None, residual=None, batchnorm=None, dropout=None, 
                 predictor_hidden_feats=None, n_tasks=None, predictor_dropout=None):
        super(pyg_Hetero_GCNPredictor, self).__init__()
        
        self.uni_trans_atoms = nn.Linear(atom_feats, hidden_feats[0])
        self.uni_trans_bonds = nn.Linear(bond_feats, hidden_feats[0])

        self.gnn = pyg_Hetero_GCN(in_feats=hidden_feats[0], hidden_feats=hidden_feats, 
                                  activation=activation, residual=residual, batchnorm=batchnorm, dropout=dropout)
        
        gnn_out_feats = hidden_feats[-1]

        self.predict = pyg_MLPLinkPredictionPredictor(gnn_out_feats, predictor_hidden_feats, n_tasks, predictor_dropout)

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        x_dict = {k:f(v) for (k,v),f in zip(x_dict.items(), [self.uni_trans_atoms, self.uni_trans_bonds])}
        
        node_feats = self.gnn(x_dict, edge_index_dict)["bonds"]
        predicted_feats = self.predict(node_feats)
        return predicted_feats
###################################################################
"""END HERE"""