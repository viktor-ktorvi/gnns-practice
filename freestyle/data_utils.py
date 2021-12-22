import torch
from torch_cluster import knn_graph
from matplotlib import pyplot as plt


def symmetric_matrix(n):
    b = torch.randn(n, n)
    return (b + b.T) / 2


def construct_adjacency_matrix_by_chance(y, p_same, p_different):
    adjacency = torch.zeros((y.shape[0], y.shape[0]), dtype=torch.int8)
    for i in range(len(y)):
        for j in range(i, len(y)):
            if i == j:
                adjacency[i, j] = 1
                continue

            if y[i] != y[j]:
                if torch.rand(1) < p_different:
                    adjacency[i, j] = 1
                    adjacency[j, i] = 1
            else:
                if torch.rand(1) < p_same:
                    adjacency[i, j] = 1
                    adjacency[j, i] = 1

    assert (adjacency == adjacency.T).all()

    return adjacency


def construct_adjacency_knn(X, k):
    adjacency = torch.zeros((X.shape[1], X.shape[1]), dtype=torch.int8)
    edge_index = knn_graph(X.T, k=k, loop=True)

    # this returns an asymmetric matrix because far away nodes will reach for nodes in a cluster but those nodes will
    # have neighbours that are closer so the friendship is not mutual :'(
    for i in range(edge_index.shape[1]):
        adjacency[edge_index[0, i], edge_index[1, i]] = 1

    # we clear friendships that are not mutual
    for i in range(adjacency.shape[0]):
        for j in range(adjacency.shape[0]):
            if adjacency[i, j] != adjacency[j, i]:
                adjacency[i, j] = 0
                adjacency[j, i] = 0

    assert (adjacency == adjacency.T).all()

    return adjacency


def add_adjacency_noise(adjacency, percentage):
    n = round(adjacency.shape[0] * percentage)

    positions = torch.randint(low=0, high=adjacency.shape[0], size=(2, n))

    for i in range(n):
        if positions[0, i] != positions[1, i]:
            adjacency[positions[0, i], positions[1, i]] ^= 1
            adjacency[positions[1, i], positions[0, i]] ^= 1

    assert (adjacency == adjacency.T).all()

    for i in range(adjacency.shape[0]):
        assert adjacency[i, i] == 1

    return adjacency


def generate_gaussian_nodes(n, y):
    mu1 = torch.randn(n, 1)
    cov1 = symmetric_matrix(n)

    mu2 = torch.randn(n, 1)
    cov2 = symmetric_matrix(n)

    mu = [mu1, mu2]
    cov = [cov1, cov2]
    nodes = torch.zeros((n, y.shape[0]))
    for i in range(len(y)):
        nodes[:, i] = torch.squeeze(mu[y[i]] + cov[y[i]] @ torch.randn(n, 1))

    return nodes


def plot_graph(nodes, y, adjacency):
    plt.figure()
    plt.axis('equal')
    plt.plot(nodes[0, y == 0], nodes[1, y == 0], 'bo', label='class 0')
    plt.plot(nodes[0, y == 1], nodes[1, y == 1], 'ro', label='class 1')
    plt.title('2D Graph')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()

    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            if adjacency[i, j] == 1:
                plt.plot([nodes[0, i], nodes[0, j]], [nodes[1, i], nodes[1, j]], color='black', alpha=0.1)

    plt.show()


def get_masks(N, train_ratio):
    train_mask = torch.zeros(N, dtype=torch.long)
    train_mask[:round(N * train_ratio)] = 1
    train_mask = train_mask[torch.randperm(len(train_mask))]
    test_mask = train_mask ^ 1

    return train_mask, test_mask
