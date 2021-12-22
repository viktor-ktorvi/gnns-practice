import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt


class GCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training, p=0.2)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


learning_rate = 0.01
weight_dacay = 5e-3
epochs = 200

if __name__ == '__main__':
    torch.manual_seed(1)
    dataset = Planetoid(root='/tmp/Cora', name='Cora')  # transform=NormalizeFeatures()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN().to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_dacay)

    loss_array = np.zeros(epochs)
    train_accuracy_array = np.zeros(epochs)
    test_accuracy_array = np.zeros(epochs)

    model.train()
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        loss_array[epoch] = loss.item()

        pred = out.argmax(dim=1)
        correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
        train_accuracy_array[epoch] = int(correct) / int(data.train_mask.sum())

        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        test_accuracy_array[epoch] = int(correct) / int(data.test_mask.sum())

    plt.figure()
    plt.title('Loss')
    plt.plot(loss_array)
    plt.xlabel('epoch [num]')
    plt.ylabel('loss')
    plt.show()

    plt.figure()
    plt.title('Accuracy')
    plt.plot(train_accuracy_array * 100, label='train')
    plt.plot(test_accuracy_array * 100, label='test')
    plt.xlabel('epoch [num]')
    plt.ylabel('accuracy [%]')
    plt.legend()
    plt.show()

    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Accuracy: {acc:.4f}')
