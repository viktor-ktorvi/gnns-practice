import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import knn_graph
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from freestyle import data_utils, gcn




if __name__ == '__main__':
    torch.manual_seed(4)

    n = 2
    N = 500

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    y = torch.randint(low=0, high=2, size=(N,))
    nodes = data_utils.generate_gaussian_nodes(n, y)

    p_same = 0.2
    p_different = 0.05

    k = 5
    adjacency = data_utils.construct_adjacency_knn(nodes, k=k)
    # adjacency = data_utils.construct_adjacency_matrix_by_chance(y, p_same, p_different)
    adjacency = data_utils.add_adjacency_noise(adjacency, percentage=0.0)

    data_utils.plot_graph(nodes, y, adjacency)

    train_ratio = 0.66
    train_mask, test_mask = data_utils.get_masks(N, train_ratio)

    train_mask = train_mask.to(device)
    test_mask = test_mask.to(device)

    y = y.to(device)
    nodes = nodes.to(device)
    adjacency = adjacency.to(device)
    model = gcn.GCN(in_features=n, num_classes=2, adjacency=adjacency).to(device)
    # model = model.to(device)

    learning_rate = 0.05
    weight_dacay = 5e-4
    epochs = 100

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_dacay)

    acc_array = np.zeros(epochs)

    model.train()
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        out = model(nodes.T)
        loss = F.nll_loss(out[train_mask == 1], y[train_mask == 1])
        loss.backward()
        optimizer.step()

        pred = out.argmax(dim=1)
        correct = (pred[train_mask == 1] == y[train_mask == 1]).sum()
        acc = int(correct) / int(train_mask.sum())

        acc_array[epoch] = acc

    plt.figure()
    plt.title('Accuracy')
    plt.plot(acc_array)
    plt.xlabel('epoch [num]')
    plt.ylabel('accuracy')
    plt.show()

    model.eval()
    pred = model(nodes.T).argmax(dim=1)
    correct = (pred[test_mask == 1] == y[test_mask == 1]).sum()
    acc = int(correct) / int(test_mask.sum())
    print(f'Accuracy: {acc:.4f}')