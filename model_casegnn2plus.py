import torch.nn as nn
from EUGATConv import EUGATConv
import torch.nn.functional as F
import torch as th
import math

class EUGATGNN(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, dropout, num_head):
        super(EUGATGNN, self).__init__()
        self.hidden_size = h_dim
        self.in_dim = in_dim
        self.EUGATConv1 = EUGATConv(in_feats=in_dim, edge_feats=in_dim, out_feats=out_dim, out_edge_feats=out_dim, num_heads=num_head)
        self.EUGATConv2 = EUGATConv(in_feats=in_dim, edge_feats=in_dim, out_feats=out_dim, out_edge_feats=out_dim, num_heads=num_head)
        self.embedding_dropout1 = nn.Dropout(dropout)
        self.embedding_dropout2 = nn.Dropout(dropout)
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

        ##Layer 1
        h = self.EUGATConv1(g, node_feats, edge_feats) 
    
        h_0 = th.squeeze(h[0]) ##h_0: node feature, h_1: edge feature
        h_1 = th.squeeze(h[1])
        h_0 = self.embedding_dropout1(h_0)
        h_1 = self.embedding_dropout2(h_1)
        h_0 = F.relu(h_0)+node_feats
        h_1 = F.relu(h_1)+edge_feats
        
        ##Layer2
        h = self.EUGATConv2(g, h_0, h_1) 

        h_0 = F.relu(h_0)
        h = th.squeeze(h[0])+node_feats
        
        ##Virtual node feature extraction as the final output
        batch_graph_embedding_index_list = g.batch_num_nodes().tolist()
        index_list = []
        num = 0
        graph_node_embedding_list = []
        for i in range(len(batch_graph_embedding_index_list)):
            index_list.append(num)
            graph_node_embedding_list.append(h[num,:])
            num += batch_graph_embedding_index_list[i]

        h = th.stack(graph_node_embedding_list)       

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
    