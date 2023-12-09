import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from torch_geometric.nn import GCNConv, SAGEConv, DNAConv, ARMAConv, ChebConv, GINConv, GatedGraphConv, SplineConv, TopKPooling, GATConv, EdgePooling, TAGConv,DynamicEdgeConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, global_sort_pool

class GCNNetAST(nn.Module):
    def __init__(self):
        super(GCNNetAST, self).__init__()

        self.embed1 = nn.Embedding(num_embeddings=5000,
                                  embedding_dim=200)

        self.conv1 = GCNConv(200, 2000)
        self.pool1 = EdgePooling(2000)
        self.conv2 = GCNConv(2000, 2000)
        self.pool2 = EdgePooling(2000)

        self.lin1 = nn.Linear(2000, 1000)
        self.lin2 = nn.Linear(1000, 4)

    def forward(self, dataGraph):
        x, edge_index = dataGraph.x.long(), dataGraph.edge_index
        batch= dataGraph.batch
        pre_x_len = len(x)
        x = self.embed1(x)
        x = x.reshape(pre_x_len,-1)

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training, p=0.3)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training, p=0.5)

        x = global_max_pool(x, batch)

        x = F.leaky_relu(self.lin1(x))
        x = F.dropout(x, training=self.training, p=0.3)
        x = F.leaky_relu(self.lin2(x))

        return x
    
class GCNNet(nn.Module):
    def __init__(self):
        super(GCNNet, self).__init__()

        self.embed1 = nn.Embedding(num_embeddings=5000,
                                  embedding_dim=200)

        self.conv1 = GCNConv(4000, 6000)
        self.pool1 = EdgePooling(6000)
        self.conv2 = GCNConv(6000, 6000)
        self.pool2 = EdgePooling(6000)

        self.lin1 = nn.Linear(6000, 3000)
        self.lin2 = nn.Linear(3000, 4)

    def forward(self, dataGraph):
        x, edge_index = dataGraph.x.long(), dataGraph.edge_index
        batch= dataGraph.batch
        pre_x_len = len(x)
        x = self.embed1(x)
        x = x.reshape(pre_x_len,-1)

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training, p=0.3)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training, p=0.5)

        x = global_max_pool(x, batch)

        x = F.leaky_relu(self.lin1(x))
        x = F.dropout(x, training=self.training, p=0.3)
        x = F.leaky_relu(self.lin2(x))

        return x

#deeptective structure
class PhpNetGraphTokensCombine(nn.Module):
    def __init__(self):
        super(PhpNetGraphTokensCombine, self).__init__()
        self.embed1 = nn.Embedding(num_embeddings=5000,
                                  embedding_dim=100)
        self.conv1 = GCNConv(2000,2000)
        self.pool1 = EdgePooling(2000)
        self.conv2 = GCNConv(2000, 4000)
        self.pool2 = EdgePooling(4000)
        
        self.embed = nn.Embedding(num_embeddings=5000,
                                  embedding_dim=100)
        self.lstm1 = nn.GRU(input_size=100,
                            hidden_size=100,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)

        self.lin1 = nn.Linear(4500, 1000)
        self.lin11 = nn.Linear(1000, 500)
        self.lin2 = nn.Linear(500, 4)

    def forward(self, dataGraph, dataTokens):
        x, edge_index = dataGraph.x.long(), dataGraph.edge_index
        batch= dataGraph.batch
        pre_x_len = len(x)
        x = self.embed1(x)
        x = x.reshape(pre_x_len,-1)
        x = self.conv1(x, edge_index)
        x, edge_index, batch,_ = self.pool1(x,edge_index,batch=dataGraph.batch)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x, edge_index, batch, _= self.pool2(x,edge_index,batch=batch)
        x = F.relu(x)

        x = global_max_pool(x, batch)

        x1 = self.embed(dataTokens)
        output1, (hidden1) = self.lstm1(x1)
        x1 = torch.cat((hidden1[0, :, :],hidden1[1, :, :],hidden1[-3, :, :], hidden1[-2, :, :], hidden1[-1, :, :]), dim=1)
        x = torch.cat([x,x1], dim=1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training, p=0.3)
        x = F.relu(self.lin11(x))
        x = F.dropout(x, training=self.training, p=0.3)
        x = F.relu(self.lin2(x))
        return x

class Vully(nn.Module):
    def __init__(self):
        super(Vully, self).__init__()
        self.embed1 = nn.Embedding(num_embeddings=5000, embedding_dim=100)
        
        # GAT 레이어 수정
        self.gat1 = GATConv(in_channels=100, out_channels=1000, heads=2, dropout=0.6)
        self.gat2 = GATConv(in_channels=1000*2, out_channels=500, heads=2, dropout=0.6)

        self.embed = nn.Embedding(num_embeddings=5000, embedding_dim=100)
        # GRU 레이어 수정
        self.lstm1 = nn.GRU(input_size=100,
                            hidden_size=100,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=False)

        self.lin1 = nn.Linear(4500, 1000)
        self.lin11 = nn.Linear(1000, 500)
        self.lin2 = nn.Linear(500, 4)

    def forward(self, dataGraph, dataTokens):
        x, edge_index = dataGraph.x.long(), dataGraph.edge_index
        batch = dataGraph.batch

        x = self.embed1(x)
        x = F.dropout(x, p=0.6, training=self.training)

        # GAT 레이어 적용
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.gat2(x, edge_index))

        x = global_max_pool(x, batch)

        x1 = self.embed(dataTokens)
        output1, (hidden1) = self.lstm1(x1)
        x1 = torch.cat((hidden1[0, :, :],hidden1[1, :, :],hidden1[-3, :, :], hidden1[-2, :, :], hidden1[-1, :, :]), dim=1)

        x = torch.cat([x, x1], dim=1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training, p=0.3)
        x = F.relu(self.lin11(x))
        x = F.dropout(x, training=self.training, p=0.3)
        x = F.relu(self.lin2(x))
        return x
