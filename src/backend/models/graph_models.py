# File: src/backend/models/graph_models.py
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def predict_link(self, z, edge_label_index):
        source_emb = z[edge_label_index[0]]
        dest_emb = z[edge_label_index[1]]
        return (source_emb * dest_emb).sum(dim=-1)
