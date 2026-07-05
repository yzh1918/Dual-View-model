# GAT_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.nn import BatchNorm

class GAT(nn.Module):
    """
    Robust GAT model for graph-level classification (one graph = one微博传播树).
    - Uses two GATConv layers with ELU activations, dropout and batchnorm.
    - Pools node representations via global_mean_pool to get graph embedding.
    - Returns logits of shape [batch_size, out_channels].
    """

    def __init__(self, in_channels, hidden_channels=128, out_channels=2, heads=8, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels=in_channels,
                             out_channels=hidden_channels,
                             heads=heads,
                             concat=True,
                             dropout=dropout)
        # batchnorm on aggregated channels
        self.bn1 = BatchNorm(hidden_channels * heads)

        # second layer reduces to hidden_channels (heads=1 so concat=False)
        self.conv2 = GATConv(in_channels=hidden_channels * heads,
                             out_channels=hidden_channels,
                             heads=1,
                             concat=False,
                             dropout=dropout)
        self.bn2 = BatchNorm(hidden_channels)

        # classifier maps pooled graph embedding -> classes
        self.classifier = nn.Linear(hidden_channels, out_channels)

        # weight init
        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier init for classifier
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, x, edge_index, batch):
        """
        x: [total_nodes_in_batch, in_channels]
        edge_index: [2, total_edges_in_batch]
        batch:  [total_nodes_in_batch] mapping nodes -> graph idx
        returns: logits [batch_size, out_channels]
        """
        # layer 1
        x = self.conv1(x, edge_index)                     # [N, hidden*heads]
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # layer 2
        x = self.conv2(x, edge_index)                     # [N, hidden]
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # graph pooling (mean)
        graph_emb = global_mean_pool(x, batch)            # [batch_size, hidden]

        logits = self.classifier(graph_emb)               # [batch_size, out_channels]
        return logits
