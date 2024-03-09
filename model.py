import torch.nn as nn
from dgl.nn import AvgPooling, EdgeGATConv
import torch.nn.functional as F
import torch as th

import math

class CaseGNN(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, dropout, num_head):
        super(CaseGNN, self).__init__()
        self.hidden_size = h_dim
        self.in_dim = in_dim
        self.EdgeGATConv1 = EdgeGATConv(in_feats=in_dim, edge_feats=in_dim, out_feats=in_dim, num_heads=num_head)
        self.EdgeGATConv2 = EdgeGATConv(in_feats=in_dim, edge_feats=in_dim, out_feats=out_dim, num_heads=num_head)
        self.avgpool = AvgPooling()
        self.embedding_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        if self.hidden_size == 0:
            stdv = 1.0 / math.sqrt(self.in_dim)
            for weight in self.parameters():
                weight.data.uniform_(-stdv, stdv)
        else:
            stdv = 1.0 / math.sqrt(self.hidden_size)
            for weight in self.parameters():
                weight.data.uniform_(-stdv, stdv)

    def forward(self, g, node_feats, edge_feats):        
        h = self.EdgeGATConv1(g, node_feats, edge_feats)
        h = th.squeeze(h)

        h = self.embedding_dropout(h)

        if self.hidden_size == 0:
            pool = self.avgpool(g, h)           
            return pool        
        else:
            h = F.relu(h)
            h = self.EdgeGATConv2(g, h, edge_feats)
            h = th.squeeze(h)
            batch_graph_embedding_index_list = g.batch_num_nodes().tolist()
            index_list = []
            num = 0
            graph_node_embedding_list = []
            for i in range(len(batch_graph_embedding_index_list)):
                index_list.append(num)
                graph_node_embedding_list.append(h[num,:])
                num += batch_graph_embedding_index_list[i]

            h = th.stack(graph_node_embedding_list)
            h = self.dropout(h)

            return h

def early_stopping(highest_f1score, epoch_f1score, epoch_num, continues_epoch):
    if epoch_f1score <= highest_f1score:
        if continues_epoch > 10:
            return [highest_f1score, True]
        else:
            continues_epoch += 1
            return [highest_f1score, False, continues_epoch]
    else:
        continues_epoch = 0
        return [epoch_f1score, False, continues_epoch]
    