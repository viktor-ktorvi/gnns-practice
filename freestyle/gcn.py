import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, psi_features, adjacency):
        super(GCNLayer, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.adjacency = adjacency
        self.c = torch.sum(adjacency, dim=0)
        self.psi_features = psi_features

        self.psi = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=15),
            nn.ReLU(),
            nn.Linear(in_features=15, out_features=psi_features),
            nn.ReLU()
        )

        self.fi = nn.Sequential(
            nn.Linear(in_features=in_features + psi_features, out_features=25),
            nn.ReLU(),
            nn.Linear(in_features=25, out_features=out_features),
            nn.ReLU()
        )

    def forward(self, nodes):
        sum_psi = 0
        psi_out = torch.zeros(nodes.shape[0], self.psi_features).to(self.device)
        for i in range(nodes.shape[0]):
            x_neighbourhood = torch.zeros((self.c[i], nodes.shape[1])).to(self.device)
            # assemble neighbours in tensor and pass throgh MLP
            neighbourhood = torch.nonzero(self.adjacency[:, i])
            for j in range(len(neighbourhood)):
                x_neighbourhood[j, :] = torch.squeeze(nodes[neighbourhood[j], :])

            psi_out[i, :] = torch.sum(self.psi(x_neighbourhood) / self.c[i], dim=0)

        fi_out = self.fi(torch.hstack((nodes, psi_out)))

        return fi_out


class GCN(nn.Module):
    def __init__(self, in_features, num_classes, adjacency):
        super(GCN, self).__init__()

        self.gcn_layer = GCNLayer(in_features=in_features, out_features=num_classes, psi_features=20,
                                  adjacency=adjacency)

    def forward(self, nodes):
        x = self.gcn_layer(nodes)

        return F.log_softmax(x, dim=1)
