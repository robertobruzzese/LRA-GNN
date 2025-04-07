import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from models.lrc_gat import LatentRelationCapturerGAT, DeepResidualGCN

class LRA_GNN(nn.Module):
    def __init__(self, in_channels=512, hidden_channels=512, num_heads=8, num_layers=4, out_channels=1, num_steps=10):
        super().__init__()
        #self.random_walk = RandomWalk(num_steps=num_steps)
        
        # MHA con GATConv (trainabile)
        self.attention = LatentRelationCapturerGAT(
            in_channels=in_channels,
            out_channels=hidden_channels,
            num_heads=num_heads
        )

        self.res_gcn = DeepResidualGCN(dim=hidden_channels, num_layers=num_layers)
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, graph, return_features=False):
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch

        head_outputs = self.attention(x, edge_index)

        deep_features = []
        for head_x in head_outputs:
            A_dense = torch.zeros((head_x.size(0), head_x.size(0)), device=head_x.device)
            A_dense[edge_index[0], edge_index[1]] = 1.0
            A_hat = DeepResidualGCN.normalize_adjacency(A_dense)

            H = self.res_gcn(A_hat, head_x)
            pooled = global_mean_pool(H, batch)
            deep_features.append(pooled)

        x = torch.stack(deep_features, dim=0).mean(dim=0)

        if return_features:
            return x
        else:
            return self.fc(x)
