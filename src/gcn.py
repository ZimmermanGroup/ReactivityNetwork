import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from dataset import *
import optuna

# Each graph in pytorch geometric is represented by a single Data object
#   --> holds all the information to describe its graph representation.
# edge_index, x (node features), y (node labels), train_mask : describes which nodes we already know their community assignments.


class GCN(torch.nn.Module):
    def __init__(self, n_features, hidden_size1, hidden_size2, model_dropout=0):
        """Defining the structure of GCN.

        Parameters
        ----------
        n_features : int
            Number of node features.
        n_layers : int
            Number of hidden convolutional layers.
        hidden_size : int
            Dimension(s) of embeddings after each hidden layer.
        model_dropout : float
            Portion of nodes to mask for updating every epoch.
        """
        torch.manual_seed(42)
        super().__init__()

        self.conv1 = GCNConv(n_features, hidden_size1)
        self.model_dropout = model_dropout

        if hidden_size2 > 0:
            self.conv2 = GCNConv(hidden_size1, hidden_size2)
            self.fc = Linear(hidden_size2, 1)
        else:
            self.conv2 = None
            self.fc = Linear(hidden_size1, 1)

        self.n_features = n_features
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.model_dropout = model_dropout

    def forward(self, X, edge_index, edge_weight):
        x = self.conv1(X, edge_index, edge_weight)
        x = F.relu(x)
        if self.model_dropout > 0:
            x = F.dropout(x, p=self.model_dropout)

        if self.conv2:
            x = self.conv2(x, edge_index, edge_weight)
            x = F.relu(x)
        if self.model_dropout > 0:
            x = F.dropout(x, p=self.model_dropout)

        x = self.fc(x)
        return x
