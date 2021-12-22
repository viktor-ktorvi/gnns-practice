import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

train_num = 500
batch_size = 32

if __name__ == '__main__':
    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    dataset = dataset.shuffle()
    train_dataset = dataset[:train_num]
    test_dataset = dataset[train_num:]

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
