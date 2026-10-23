"""GraphSAGE graph classifier for propagation-tree fake news detection."""
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_max_pool, global_mean_pool
from torch_geometric.utils import to_undirected


class SAGENet(torch.nn.Module):
    """Two SAGEConv layers, mean+max pooling over nodes, linear classifier.

    Edges are made undirected so information flows both up and down the tree.
    """

    def __init__(self, in_dim, hid=128, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hid)
        self.conv2 = SAGEConv(hid, hid)
        self.lin = torch.nn.Linear(hid * 2, 2)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        ei = to_undirected(edge_index)
        x = F.relu(self.conv1(x, ei))
        x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.conv2(x, ei))
        x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        x = F.dropout(x, self.dropout, self.training)
        return self.lin(x)
