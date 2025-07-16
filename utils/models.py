import os
import random
import torch
import mne

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import random_split, TensorDataset
from torch_geometric.utils import to_undirected
from scipy.spatial.distance import cdist
from scipy.stats import skew, kurtosis
from tqdm import tqdm


class SimpleGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, fc_hidden, num_classes, n_drop):
        '''
        Args:
            in_channels: dimension of input
            hidden_channels: numbers of hidden_layer
            fc_hidden: dimension of input
            num_classes: numbers of classification
        Note:
            here trial will be packed as batch
        '''
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)  
        self.fc1 = nn.Linear(hidden_channels, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, fc_hidden)
        self.classifier = nn.Linear(fc_hidden, num_classes)
        self.dropout = nn.Dropout(n_drop) 
        
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        x = global_mean_pool(x, batch=batch)
        x = self.dropout(x)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.classifier(x)
        
        return x
                
                
class SimpleCCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, out_channel, n_drop):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=2)
        self.conv2 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(n_drop)

        self.fc1 = nn.LazyLinear(256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, out_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
    


