import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid

# 데이터셋 로드
dataset = Planetoid(root='/path/to/dataset', name='Cora')

# 데이터 로더 생성
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# GNN 모델 정의
class GNNModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

# 모델 초기화
num_features = dataset.num_features
num_classes = dataset.num_classes
model = GNNModel(num_features, num_classes)

# 옵티마이저 정의
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 학습 함수
def train():
    model.train()
    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()

# 테스트 함수
def test():
    model.eval()
    correct = 0
    for data in loader:
        out = model(data)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(dataset)

# 학습 및 테스트 실행
for epoch in range(1, 101):
    train()
    accuracy = test()
    print(f'Epoch: {epoch}, Accuracy: {accuracy:.4f}')
