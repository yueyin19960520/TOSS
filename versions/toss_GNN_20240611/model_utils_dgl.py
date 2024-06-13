import torch
import dgl
import dgllife
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
#from dgl.nn.pytorch import GraphConv
#from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch import edge_softmax
#from dgl.nn.functional import edge_softmax
from dgl.utils import expand_as_pair
from dgl.nn.pytorch import HeteroGraphConv
from dgl.nn.pytorch.utils import Identity


#########################################################################################################################
################################################## NODE CLASSIFICATION ##################################################
#########################################################################################################################
class dgl_MLPNodeClassificationPredictor(nn.Module):

    def __init__(self, in_feats, hidden_feats, n_tasks, dropout=0.):
        super(dgl_MLPNodeClassificationPredictor, self).__init__()

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
class dgl_GraphConv(nn.Module):
    def __init__(self, in_feats, out_feats, norm='both',activation=None):
        super(dgl_GraphConv, self).__init__()

        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm

        self.lin_weight = nn.Linear(in_feats, out_feats, bias=True)
        self.rot_weight = nn.Linear(in_feats, out_feats, bias=True)

        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        torch.nn.init.ones_(self.lin_weight.weight)
        torch.nn.init.ones_(self.rot_weight.weight)
        
        torch.nn.init.zeros_(self.lin_weight.bias)
        torch.nn.init.zeros_(self.rot_weight.bias)

    def forward(self, graph, feat, weight=None, edge_weight=None):
        with graph.local_scope():

            aggregate_fn = fn.copy_src('h', 'm')
            feat_src, feat_dst = expand_as_pair(feat, graph)

            # aggregate first then mult W
            graph.srcdata['h'] = feat_src
            graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']
            
            rst = self.lin_weight(rst)
            dst = self.rot_weight(feat_dst)
            rst += dst
                
            if self._activation is not None:
                rst = self._activation(rst)

            return rst


class dgl_GCNLayer(nn.Module):
    
    def __init__(self, in_feats, out_feats, gnn_norm, activation, 
                 residual, batchnorm, dropout):
        super(dgl_GCNLayer, self).__init__()

        self.activation = activation

        self.graph_conv = dgl_GraphConv(in_feats=in_feats, out_feats=out_feats, norm=gnn_norm, activation=activation)

        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats, bias = True) #default = True

        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)
            
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        self.graph_conv.reset_parameters()

        if self.residual:
            self.res_connection.reset_parameters()

        if self.bn:
            self.bn_layer.reset_parameters()

    def forward(self, g, feats):
        new_feats = self.graph_conv(g, feats)
    
        if self.residual:
            res_feats = self.activation(self.res_connection(feats))
            new_feats = new_feats + res_feats

        new_feats = self.dropout(new_feats)

        if self.bn:
            new_feats = self.bn_layer(new_feats)

        return new_feats


class dgl_GCN(nn.Module):
    
    def __init__(self, in_feats, hidden_feats=None, gnn_norm=None, activation=None, 
                 residual=None, batchnorm=None, dropout=None):
        super(dgl_GCN, self).__init__()

        n_layers = len(hidden_feats)
        
        if gnn_norm is None:
            gnn_norm = ['none' for _ in range(n_layers)]
        if activation is None:
            activation = [F.relu for _ in range(n_layers)]

        if residual is None:
            residual = [True for _ in range(n_layers)]
        if batchnorm is None:
            batchnorm = [True for _ in range(n_layers)]
        if dropout is None:
            dropout = [0. for _ in range(n_layers)]

        self.hidden_feats = hidden_feats
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(dgl_GCNLayer(in_feats, hidden_feats[i], gnn_norm[i], activation[i], 
                                               residual[i], batchnorm[i], dropout[i]))
            in_feats = hidden_feats[i]

    def reset_parameters(self):
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, g, feats):
        for gnn in self.gnn_layers:
            feats = gnn(g, feats)
        return feats


class dgl_GCNPredictor(nn.Module):
    
    def __init__(self, in_feats, hidden_feats=None, 
                 gnn_norm=None, activation=None, residual=None, batchnorm=None, dropout=None, 
                 predictor_hidden_feats=None, n_tasks=None, predictor_dropout=None):
        super(dgl_GCNPredictor, self).__init__()

        self.gnn = dgl_GCN(in_feats=in_feats, hidden_feats=hidden_feats, gnn_norm=gnn_norm, activation=activation,
                          residual=residual, batchnorm=batchnorm, dropout=dropout)
        
        gnn_out_feats = self.gnn.hidden_feats[-1]

        self.predict = dgl_MLPNodeClassificationPredictor(gnn_out_feats, predictor_hidden_feats, n_tasks, predictor_dropout)

    def forward(self, bg, feats):
        node_feats = self.gnn(bg, feats)
        predicted_feats = self.predict(node_feats)
        return predicted_feats


#################################################
# GAT(GATLayer) + MLPPredictor --> GCNPredictor #
#################################################
class dgl_GATConv(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, feat_drop=0., attn_drop=0., negative_slope=0.2,
                 residual=False, activation=None,bias=True):
        super(dgl_GATConv, self).__init__()
        self.num_heads = num_heads
        self.in_feats = in_feats
        self.out_feats = out_feats

        self.lin = nn.Linear(self.in_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        
        dropout = attn_drop
        self.dropout = nn.Dropout(dropout)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_feats,)))
            
        if residual:
            #if self.in_feats != out_feats * num_heads:
            self.res_fc = nn.Linear(self.in_feats, num_heads * out_feats, bias=False)
            #else:
                #self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
            
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.lin.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():
            h_src = h_dst = feat
            
            feat_src = feat_dst = self.lin(h_src).view(-1, self.num_heads, self.out_feats)

            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)

            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})

            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))

            graph.edata['a'] = self.dropout(edge_softmax(graph, e))

            graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']

            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(-1, self.num_heads, self.out_feats)
                rst = rst + resval

            if self.bias is not None:
                rst = rst + self.bias.view(-1, self.num_heads, self.out_feats)

            if self.activation:
                rst = self.activation(rst)

            return rst


class dgl_GATLayer(nn.Module):

    def __init__(self, in_feats, out_feats, num_heads, feat_drop, attn_drop, alpha, residual, bias,
                 agg_mode, activation):
        super(dgl_GATLayer, self).__init__()

        self.activation = activation

        self.gat_conv = dgl_GATConv(in_feats=in_feats, out_feats=out_feats, num_heads=num_heads, 
                                feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=alpha, residual=residual, bias=bias)

        #residual already in the GATConv!!!
        #attention dropout already in the GATConv!!!

        self.agg_mode = agg_mode

        self.bn = True
        if self.bn:
            if self.agg_mode == 'flatten':
                self.bn_layer = nn.BatchNorm1d(out_feats * num_heads)
            else:
                self.bn_layer = nn.BatchNorm1d(out_feats)


    def reset_parameters(self):
        self.gat_conv.reset_parameters()

        if self.bn:
            self.bn_layer_flatten.reset_parameters()
            self.bn_layer_mean.reset_parameters()

    def forward(self, g, feats):
        new_feats = self.gat_conv(g, feats)

        if self.agg_mode == 'flatten':
            new_feats = new_feats.flatten(1)
        else:
            new_feats = new_feats.mean(1)

        if self.activation is not None:
            new_feats = self.activation(new_feats)

        if self.bn:
            if self.agg_mode == 'flatten':
                new_feats = self.bn_layer(new_feats)
            else:
                new_feats = self.bn_layer(new_feats)

        return new_feats

class dgl_GAT(nn.Module):
    
    def __init__(self, in_feats, hidden_feats=None, num_heads=None, feat_drops=None, attn_drops=None, alphas=None, residuals=None, biases=None,
                 agg_modes=None, activations=None):
        super(dgl_GAT, self).__init__()

        n_layers = len(hidden_feats)

        if agg_modes is None:
            agg_modes = ['flatten' for _ in range(n_layers - 1)]
            agg_modes.append('mean')
        if activations is None:
            activations = [F.elu for _ in range(n_layers - 1)]
            activations.append(None)

        if num_heads is None:
            num_heads = [4 for _ in range(n_layers)]
        if feat_drops is None:
            feat_drops = [0. for _ in range(n_layers)]
        if attn_drops is None:
            attn_drops = [0. for _ in range(n_layers)]
        if alphas is None:
            alphas = [0.2 for _ in range(n_layers)]
        if residuals is None:
            residuals = [True for _ in range(n_layers)]
        if biases is None:
            biases = [True for _ in range(n_layers)]
            
        self.hidden_feats = hidden_feats
        self.num_heads = num_heads
        self.agg_modes = agg_modes
        self.gnn_layers = nn.ModuleList()

        for i in range(n_layers):
            self.gnn_layers.append(dgl_GATLayer(in_feats, hidden_feats[i], num_heads[i],feat_drops[i], attn_drops[i], alphas[i], residuals[i], biases[i],
                                               agg_modes[i], activations[i]))
            if agg_modes[i] == 'flatten':
                in_feats = hidden_feats[i] * num_heads[i]
            else:
                in_feats = hidden_feats[i]

    def reset_parameters(self):
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, g, feats):
        for gnn in self.gnn_layers:
            feats = gnn(g, feats)
        return feats


class dgl_GATPredictor(nn.Module):
    def __init__(self, in_feats, hidden_feats=None, num_heads=None, feat_drops=None, attn_drops=None, alphas=None, residuals=None, biases=None,
                 agg_modes=None, activations=None,
                 predictor_hidden_feats=None, n_tasks=None, predictor_dropout=None):
        super(dgl_GATPredictor, self).__init__()

        self.gnn = dgl_GAT(in_feats=in_feats, hidden_feats=hidden_feats, num_heads=num_heads, 
                          feat_drops=feat_drops, attn_drops=attn_drops, alphas=alphas, residuals=residuals, biases=biases,
                          agg_modes=agg_modes, activations=activations)

        if self.gnn.agg_modes[-1] == 'flatten': #the difference is the agg_mode.
            gnn_out_feats = self.gnn.hidden_feats[-1] * self.gnn.num_heads[-1]
        else:
            gnn_out_feats = self.gnn.hidden_feats[-1]

        self.predict = dgl_MLPNodeClassificationPredictor(gnn_out_feats, predictor_hidden_feats, n_tasks, predictor_dropout)

    def forward(self, bg, feats):
        node_feats = self.gnn(bg, feats)
        predicted_feates = self.predict(node_feats)
        return predicted_feates


#####################################################
#AttentiveFP + MLPPredictor --> AttentiveFPPredictor#
#####################################################
class dgl_AttentiveGRU1(nn.Module):
    
    def __init__(self, node_feat_size, edge_feat_size, edge_hidden_size, dropout):
        super(dgl_AttentiveGRU1, self).__init__()

        self.edge_transform = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(edge_feat_size, edge_hidden_size))   #new added linear transform.
            
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def reset_parameters(self):
        self.edge_transform[1].reset_parameters()
        self.gru.reset_parameters()

    def forward(self, g, edge_logits, edge_feats, node_feats):
        g = g.local_var()
        g.edata['e'] = edge_softmax(g, edge_logits) * self.edge_transform(edge_feats) #(actually a_vu * W[h_u]) 
        g.update_all(fn.copy_edge('e', 'm'), fn.sum('m', 'c')) #(copy edge data to "m", sum all "m" data to "c")
        context = F.elu(g.ndata['c'])   #elu activate, get the c_v
        return F.relu(self.gru(context, node_feats)) #(actually, gru(c_v,h_v), and after relu, it is the h_v new.)

class dgl_AttentiveGRU2(nn.Module):

    def __init__(self, node_feat_size, edge_hidden_size, dropout):
        super(dgl_AttentiveGRU2, self).__init__()

        self.project_node = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(node_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def reset_parameters(self):
        self.project_node[1].reset_parameters()
        self.gru.reset_parameters()

    def forward(self, g, edge_logits, node_feats):
        g = g.local_var()
        g.edata['a'] = edge_softmax(g, edge_logits)
        g.ndata['hv'] = self.project_node(node_feats)

        g.update_all(fn.src_mul_edge('hv', 'a', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))

class dgl_GetContext(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, graph_feat_size, dropout):
        super(dgl_GetContext, self).__init__()

        self.project_node = nn.Sequential(nn.Linear(node_feat_size, graph_feat_size),nn.LeakyReLU())
        self.project_edge1 = nn.Sequential(nn.Linear(node_feat_size + edge_feat_size, graph_feat_size),nn.LeakyReLU())
        self.project_edge2 = nn.Sequential(nn.Dropout(dropout),nn.Linear(2 * graph_feat_size, 1),nn.LeakyReLU())
        self.attentive_gru = dgl_AttentiveGRU1(graph_feat_size, graph_feat_size, graph_feat_size, dropout)

    def reset_parameters(self):
        self.project_node[0].reset_parameters()
        self.project_edge1[0].reset_parameters()
        self.project_edge2[1].reset_parameters()
        self.attentive_gru.reset_parameters()

    def apply_edges1(self, edges):
        return {'he1': torch.cat([edges.src['hv'], edges.data['he']], dim=1)}

    def apply_edges2(self, edges):
        return {'he2': torch.cat([edges.dst['hv_new'], edges.data['he1']], dim=1)}

    def forward(self, g, node_feats, edge_feats):
        g = g.local_var()
        g.ndata['hv'] = node_feats #assign data
        g.ndata['hv_new'] = self.project_node(node_feats) #linear transform    (actually, h_v).
        g.edata['he'] = edge_feats #assign data

        g.apply_edges(self.apply_edges1)  #he1 = concat(node['hv'],edge['he'])
        g.edata['he1'] = self.project_edge1(g.edata['he1']) #linear transform    (actually, h_u).
        g.apply_edges(self.apply_edges2)  #he2 = concat(node['hv_new'],edge['he1'])     (actually [h_v, h_u])
        logits = self.project_edge2(g.edata['he2']) #(actually, get e_vu=W[h_v, h_u], but without the leaky_relu).  

        return self.attentive_gru(g, logits, g.edata['he1'], g.ndata['hv_new']) #actually, it is the h_v new!!!!!

class dgl_GNNLayer(nn.Module):

    def __init__(self, node_feat_size, graph_feat_size, dropout):
        super(dgl_GNNLayer, self).__init__()

        self.project_edge = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * node_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = dgl_AttentiveGRU2(node_feat_size, graph_feat_size, dropout)

    def reset_parameters(self):
        self.project_edge[1].reset_parameters()
        self.attentive_gru.reset_parameters()

    def apply_edges(self, edges):
        return {'he': torch.cat([edges.dst['hv'], edges.src['hv']], dim=1)}

    def forward(self, g, node_feats):
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.apply_edges(self.apply_edges)
        logits = self.project_edge(g.edata['he'])

        return self.attentive_gru(g, logits, node_feats)
    

class dgl_AFPPredictor(nn.Module):
    def __init__(self, node_feat_size=None, edge_feat_size=None, graph_feat_size=None, dropout=None,
                 num_layers=None,
                 predictor_hidden_feats=None, n_tasks=None, predictor_dropout=None):
        super(dgl_AFPPredictor, self).__init__()

        self.init_context = dgl_GetContext(node_feat_size, edge_feat_size, graph_feat_size, dropout) #work as the first layer

        self.gnn_layers = nn.ModuleList()

        for _ in range(num_layers - 1):
            self.gnn_layers.append(dgl_GNNLayer(graph_feat_size, graph_feat_size, dropout))
        
        self.predict = dgl_MLPNodeClassificationPredictor(graph_feat_size, predictor_hidden_feats, n_tasks, predictor_dropout)


    def reset_parameters(self):
        self.init_context.reset_parameters()

        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.init_context(g, node_feats, edge_feats)

        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats)

        predicted_feats = self.predict(node_feats)
        return predicted_feats


#########################################
# MPNN + MLPPredictor --> MPNNPredictor #
#########################################
class dgl_NNConv(nn.Module):

    def __init__(self, in_feats, out_feats, edge_func, aggregator_type=None, residual=False, bias=True):
        super(dgl_NNConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.edge_func = edge_func
 
        self.reducer = fn.max if aggregator_type == 'max' else fn.mean
    
        self.bn = True
        if self.bn:
            self.bn_layer = nn.BatchNorm1d(out_feats)
        
        if residual:
            self.res_fc = nn.Linear(self._in_dst_feats, out_feats, bias=False)
        else:
            self.register_buffer('res_fc', None)
            
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_buffer('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        #return the recommended gain value for the given nonlinear function
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        if self.bn:
            self.bn_layer.reset_parameters()

    def forward(self, graph, feat, efeat):
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata['h'] = feat_src.unsqueeze(-1)    
            #(V, node_out_feats, 1)
            temp = self.edge_func(efeat)                   
            #(W, edge_in_feats)-->(W, node_out_feats**2)
            graph.edata['w'] = temp.view(-1, self._in_src_feats, self._out_feats)
            #(W,node_out_feats**2)-->(W,node_out_feats,node_out_feats)
            graph.update_all(fn.u_mul_e('h', 'w', 'm'), self.reducer('m', 'neigh'))
            #(V, node_out_feats, 1)*(W,node_out_feats,node_out_feats)
            rst = graph.dstdata['neigh'].sum(dim=1) # (n, d_out)

            if self.res_fc is not None:
                rst = rst + self.res_fc(feat_dst)

            if self.bias is not None:
                rst = rst + self.bias
            
            if self.bn:
                rst = self.bn_layer(rst)
            
            return rst
        

class dgl_MPNNGNN(nn.Module):

    def __init__(self, node_in_feats, edge_in_feats, node_out_feats=None, edge_hidden_feats=None, num_step_message_passing=None):
        
        super(dgl_MPNNGNN, self).__init__()

        self.project_node_feats = nn.Sequential(nn.Linear(node_in_feats, node_out_feats),
                                                nn.ReLU())
        
        self.num_step_message_passing = num_step_message_passing
        
        edge_network = nn.Sequential(nn.Linear(edge_in_feats, edge_hidden_feats),
                                     nn.ReLU(),
                                     nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats))
        
        self.gnn_layer = dgl_NNConv(in_feats=node_out_feats, out_feats=node_out_feats, edge_func=edge_network, aggregator_type='mean')
        
        self.gru = nn.GRU(node_out_feats, node_out_feats)

    def reset_parameters(self):
        self.project_node_feats[0].reset_parameters()
        self.gnn_layer.reset_parameters()
        
        for layer in self.gnn_layer.edge_func:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

        self.gru.reset_parameters()

    def forward(self, g, node_feats, edge_feats):

        node_feats = self.project_node_feats(node_feats) # (V, node_in_feats) --> (V, node_out_feats)
        hidden_feats = node_feats.unsqueeze(0)           # (V, node_out_feats) --> (1, V, node_out_feats)
                                                         # edge_feats = (W, edge_in_feats)
        for _ in range(self.num_step_message_passing):   #just times 
            node_feats = F.relu(self.gnn_layer(g, node_feats, edge_feats))
            node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)
            node_feats = node_feats.squeeze(0)

        return node_feats


class dgl_MPNNPredictor(nn.Module):

    def __init__(self, node_in_feats=None, edge_in_feats=None, node_out_feats=None, edge_hidden_feats=None, num_step_message_passing=None,
                 n_tasks=None, predictor_hidden_feats=None, predictor_dropout=None):
        
        super(dgl_MPNNPredictor, self).__init__()

        self.gnn = dgl_MPNNGNN(node_in_feats=node_in_feats, node_out_feats=node_out_feats, edge_in_feats=edge_in_feats, edge_hidden_feats=edge_hidden_feats,
                              num_step_message_passing=num_step_message_passing)
        
        self.predict = dgl_MLPNodeClassificationPredictor(node_out_feats, predictor_hidden_feats, n_tasks, predictor_dropout)

    def forward(self, g, node_feats, edge_feats):

        node_feats = self.gnn(g, node_feats, edge_feats)
        predicted_feats = self.predict(node_feats)
        return predicted_feats


#################################################################################################################################
################################################## LINK PREDICTION ##### HOMO ###################################################
#################################################################################################################################
class dgl_Homo_MLPLinkPredictionPredictor(nn.Module):

    def __init__(self, node_feats):
        super().__init__()

        self.predict = nn.Sequential(
            nn.Linear(node_feats*2, node_feats),
            nn.ReLU(),
            nn.Linear(node_feats, 2))

    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.predict(h)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']


class dgl_Homo_DPLinkPredictionPredictor(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['LP'] = h
            graph.apply_edges(fn.u_dot_v('LP', 'LP', 'score'))
            return graph.edata['score']


################################################################
# GCN + Homo_LinkPredictionPredictor --> Homo_GCNLinkPredictor #
################################################################
class dgl_Homo_GCNLinkPredictor(nn.Module):
    
    def __init__(self, in_feats, hidden_feats=None, 
                 gnn_norm=None, activation=None, residual=None, batchnorm=None, dropout=None,
                 predictor=None,):

        super(dgl_Homo_GCNLinkPredictor, self).__init__()

        self.gnn = dgl_GCN(in_feats=in_feats, hidden_feats=hidden_feats, gnn_norm=gnn_norm, activation=activation,
                          residual=residual, batchnorm=batchnorm, dropout=dropout)
        
        gnn_out_feats = self.gnn.hidden_feats[-1]

        if predictor == "DOT":
            self.predict = dgl_Homo_DPLinkPredictionPredictor()
        else:
            self.predict = dgl_Homo_MLPLinkPredictionPredictor(gnn_out_feats)

    def forward(self, bg, feats):
        node_feats = self.gnn(bg, feats)
        edge_feats = self.predict(bg, node_feats)
        return edge_feats


################################################################
# GAT + Homo_LinkPredictionPredictor --> Homo_GATLinkPredictor #
################################################################
class dgl_Homo_GATLinkPredictor(nn.Module):

    def __init__(self, in_feats, hidden_feats=None, num_heads=None, feat_drops=None, attn_drops=None, alphas=None, residuals=None, biases=None,
                 agg_modes=None, activations=None,
                 predictor=None,):

        super(dgl_Homo_GATLinkPredictor, self).__init__()

        self.gnn = dgl_GAT(in_feats=in_feats, hidden_feats=hidden_feats, num_heads=num_heads, 
                          feat_drops=feat_drops, attn_drops=attn_drops, alphas=alphas, residuals=residuals, biases=biases,
                          agg_modes=agg_modes, activations=activations)

        if self.gnn.agg_modes[-1] == 'flatten': #the difference is the agg_mode.
            gnn_out_feats = self.gnn.hidden_feats[-1] * self.gnn.num_heads[-1]
        else:
            gnn_out_feats = self.gnn.hidden_feats[-1]

        if predictor == "DOT":
            self.predict = dgl_Homo_DPLinkPredictionPredictor()
        else:
            self.predict = dgl_Homo_MLPLinkPredictionPredictor(gnn_out_feats)

    def forward(self, bg, feats):
        node_feats = self.gnn(bg, feats)
        edge_feates = self.predict(bg, node_feats)
        return edge_feates


################################################################
# AFP + Homo_LinkPredictionPredictor --> Homo_AFPLinkPredictor #
################################################################
class dgl_Homo_AFPLinkPredictor(nn.Module):

    def __init__(self, node_feat_size=None, edge_feat_size=None, graph_feat_size=None, dropout=None,
                 num_layers=None,
                 predictor=None):

        super(dgl_Homo_AFPLinkPredictor, self).__init__()

        self.init_context = dgl_GetContext(node_feat_size, edge_feat_size, graph_feat_size, dropout) #work as the first layer

        self.gnn_layers = nn.ModuleList()

        for _ in range(num_layers - 1):
            self.gnn_layers.append(dgl_GNNLayer(graph_feat_size, graph_feat_size, dropout))

        if predictor == "DOT":
            self.predict = dgl_Homo_DPLinkPredictionPredictor()
        else:
            self.predict = dgl_Homo_MLPLinkPredictionPredictor(graph_feat_size)

    def reset_parameters(self):
        self.init_context.reset_parameters()

        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    """
    def forward(self, bg, neg_bg, node_feats, edge_feats):
        node_feats = self.init_context(bg, node_feats, edge_feats)

        for gnn in self.gnn_layers:
            node_feats = gnn(bg, node_feats)

        edge_feats = self.predict(neg_bg, node_feats)
        return edge_feats

    """
    def forward(self, bg, node_feats, edge_feats):
        node_feats = self.init_context(bg, node_feats, edge_feats)

        for gnn in self.gnn_layers:
            node_feats = gnn(bg, node_feats)

        edge_feats = self.predict(bg, node_feats)
        return edge_feats
    

##################################################################
# MPNN + Homo_LinkPredictionPredictor --> Homo_MPNNLinkPredictor #
##################################################################
class dgl_Homo_MPNNLinkPredictor(nn.Module):

    def __init__(self, node_in_feats=None, edge_in_feats=None, node_out_feats=None, edge_hidden_feats=None, num_step_message_passing=None,
                 predictor=None):
        
        super(dgl_Homo_MPNNLinkPredictor, self).__init__()

        self.gnn = dgl_MPNNGNN(node_in_feats=node_in_feats, node_out_feats=node_out_feats, edge_in_feats=edge_in_feats, edge_hidden_feats=edge_hidden_feats,
                              num_step_message_passing=num_step_message_passing)
        
        if predictor == "DOT":
            self.predict = dgl_Homo_DPLinkPredictionPredictor()
        else:
            self.predict = dgl_Homo_MLPLinkPredictionPredictor(node_out_feats)

    def forward(self, bg, node_feats, edge_feats):

        node_feats = self.gnn(bg, node_feats, edge_feats)
        edge_feats = self.predict(bg, node_feats)

        return edge_feats


#################################################################################################################################
################################################## LINK PREDICTION ##### HETERO #################################################
#################################################################################################################################
class dgl_Hetero_MLPLinkPredictionPredictor(nn.Module):

    def __init__(self, in_feats, hidden_feats, n_tasks, dropout=0.):
        super(dgl_MLPNodeClassificationPredictor, self).__init__()

        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_feats),
            nn.Linear(hidden_feats, n_tasks)
        )

    def forward(self, feats):
        return self.predict(feats)


###########################################################################
# Hetero_GCN + Hetero_LinkPredictionPredictor --> Hetero_GCNLinkPredictor #
###########################################################################
class dgl_Hetero_GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, rel_names, gnn_norm, activation, residual, batchnorm, dropout):
        super(dgl_Hetero_GCNLayer,self).__init__()
        self.activation = activation
        
        self.graph_conv = HeteroGraphConv({rel:dgl_GraphConv(in_feats, out_feats, norm=gnn_norm, activation=activation) for rel in rel_names}, aggregate="sum")
        
        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats, bias=True)
            
        self.bn = batchnorm
        if batchnorm:
            self.bn_layer_1 = nn.BatchNorm1d(out_feats)
            self.bn_layer_2 = nn.BatchNorm1d(out_feats)
            
        self.dropout = nn.Dropout(dropout)
        
    def reset_parameters(self):
        self.graph_conv.reset_parameters()
        
        if self.residual:
            self.res_connection.reset_parameters()
        if self.bn:
            self.bn_layer_1.reset_parameters()
            self.bn_layer_2.reset_parameters()
            
    def forward(self, graph, feats):
        new_feats = self.graph_conv(graph,feats)
        
        if self.residual:
            res_feats = {k: self.activation(self.res_connection(v)) for k,v in feats.items()}
            new_feats = {k: v + res_feats[k] for k,v in new_feats.items()}
            
        new_feats = {k: self.dropout(v) for k,v in new_feats.items()}
        
        if self.bn:
            #new_feats = {k: self.bn_layer(v) for k,v in new_feats.items()}
            new_feats = {k: f(v) for (k,v),f in zip(new_feats.items(), [self.bn_layer_1, self.bn_layer_2])}
        return new_feats
    
    
class dgl_Hetero_GCN(nn.Module):
    def __init__(self,in_feats=None, hidden_feats=None, Etypes=None,
                 gnn_norm=None, activation=None, residual=None, batchnorm=None, dropout=None):
        super(dgl_Hetero_GCN, self).__init__()
        
        n_layers = len(hidden_feats)
        
        if gnn_norm is None:
            gnn_norm = ['none' for _ in range(n_layers)]
        if activation is None:
            activation = [F.relu for _ in range(n_layers)]
        if batchnorm is None:
            batchnorm = [True for _ in range(n_layers)]
        if residual is None:
            residual = [True for _ in range(n_layers)]
        if dropout is None:
            dropout = [0. for _ in range(n_layers)]

        self.hidden_feats = hidden_feats
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(dgl_Hetero_GCNLayer(in_feats, hidden_feats[i], Etypes,
                                                      gnn_norm[i], activation[i], residual[i], batchnorm[i],dropout[i]))
            in_feats = hidden_feats[i]
            
    def reset_parameters(self):
        for gnn in self.gnn_layers:
            gnn.reset_parameters()
    
    def forward(self, g, feats):
        for gnn in self.gnn_layers:
            feats = gnn(g, feats)
        return feats
    

class dgl_Hetero_GCNPredictor(nn.Module):
    def __init__(self, atom_feats=None, bond_feats=None, hidden_feats=None, Etypes=None,
                 gnn_norm=None, activation=None, residual=None, batchnorm=None, dropout=None,
                 predictor_hidden_feats=None, n_tasks=None,predictor_dropout=None):
        super(dgl_Hetero_GCNPredictor, self).__init__()
        
        self.uni_trans_atoms = nn.Linear(atom_feats, hidden_feats[0])  #?????????????????????????????????????????????????????????????
        self.uni_trans_bonds = nn.Linear(bond_feats, hidden_feats[0])   #?????????????????????????????????????????????????????????????
        
        self.gnn = dgl_Hetero_GCN(in_feats=hidden_feats[0], hidden_feats=hidden_feats, Etypes=Etypes,
                                 gnn_norm=gnn_norm, activation=activation, residual=residual, batchnorm=batchnorm,dropout=dropout)
        
        gnn_out_feats = self.gnn.hidden_feats[-1]
        
        self.predict = dgl_MLPLinkPredictionPredictor(gnn_out_feats, predictor_hidden_feats, n_tasks, predictor_dropout)
        
    def forward(self, bg, feats):
        feats = {k:f(v) for (k,v),f in zip(feats.items(),[self.uni_trans_atoms, self.uni_trans_bonds])}     #?????????????????????????????????????????????????????
        new_feats = self.gnn(bg, feats)["bonds"]
        predicted_feats = self.predict(new_feats)
        return predicted_feats
"""END HERE"""
"""
class old_dgl_GraphConv(nn.Module):
    def __init__(self, in_feats, out_feats, norm='both', weight=True, bias=True, activation=None):
        super(old_dgl_GraphConv, self).__init__()

        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm

        if weight:
            self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, graph, feat, weight=None, edge_weight=None):
        with graph.local_scope():

            aggregate_fn = fn.copy_src('h', 'm')
            feat_src, feat_dst = expand_as_pair(feat, graph)

            weight = self.weight
            # aggregate first then mult W
            graph.srcdata['h'] = feat_src
            graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']
            rst = torch.matmul(rst, weight)

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst
"""