import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from torch_geometric.nn import GCNConv, SAGEConv, DNAConv, ARMAConv, ChebConv, GINConv, GatedGraphConv, SplineConv, TopKPooling, GATConv, EdgePooling, TAGConv,DynamicEdgeConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, global_sort_pool

class GCNNet(nn.Module):
    def __init__(self):
        super(GCNNet, self).__init__()

        self.embed1 = nn.Embedding(num_embeddings=5000,
                                  embedding_dim=200)

        self.conv1 = GCNConv(4000, 4000)
        #self.pool1 = EdgePooling(4000)
        self.conv2 = GCNConv(4000, 4000)
        #self.pool2 = EdgePooling(4000)
        #self.conv3 = GCNConv(6000, 6000)
        #self.pool3 = EdgePooling(6000)

        self.lin1 = nn.Linear(4000, 1000)
        self.lin2 = nn.Linear(1000, 4)

    def forward(self, dataGraph, dataTokens=None):
        x, edge_index = dataGraph.x.long(), dataGraph.edge_index
        batch= dataGraph.batch
        pre_x_len = len(x)
        x = self.embed1(x)
        x = x.reshape(pre_x_len,-1)

        x = self.conv1(x, edge_index)
        #x, edge_index, batch,_ = self.pool1(x,edge_index,batch=dataGraph.batch)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.3)

        x = self.conv2(x, edge_index)
        #x, edge_index, batch, _= self.pool2(x,edge_index,batch=batch)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.5)

        #x = self.conv3(x, edge_index)
        #x, edge_index, batch, _= self.pool3(x, edge_index, batch=batch)
        #x = F.relu(x)

        x = global_max_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training, p=0.3)
        x = F.relu(self.lin2(x))

        return x