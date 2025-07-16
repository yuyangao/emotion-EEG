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


def train(model, model_name, train_loader, optimizer, criterion):
    model.train()
    tol_loss = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        ## zero gradient
        optimizer.zero_grad()
        ## forward pass
        if model_name == 'GCN':
            out = model.forward(batch.x, batch.edge_index, batch.batch)
            labels = batch.y
        elif model_name == 'CNN':
            inputs, labels = batch
            out = model.forward(inputs)
        ## calculate loss according to output and y
        loss = criterion(out, labels)
        ## backward propagation
        loss.backward()
        ## do optimization
        optimizer.step()
        ## accumulate loss
        tol_loss += loss.item()
    ## return the mean loss    
    return tol_loss / len(train_loader)

def test(model, model_name, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    ## no need to do gradient descent in test phase  
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            ## forward pass
            if model_name == 'GCN':
                out = model.forward(batch.x, batch.edge_index, batch.batch)
                labels = batch.y
            elif model_name == 'CNN':
                inputs, labels = batch
                out = model.forward(inputs)
            prediction = out.argmax(dim = 1)
            correct += (prediction == labels).sum().item()
            total += labels.size(0)
            
        accuracy = correct / total
    return  accuracy   

def fit(model, model_name,  train_loader, test_loader, epochs = 200, lr = 0.01):
        ## define optimizer and criterion
        # L2 penalty
        optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=1e-3)
        ## here i choose cross entropy
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
                train_loss = train(model, model_name, train_loader, optimizer, criterion)  
                test_acc = test(model, model_name, test_loader)
                print(f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.4f}')
        
        return test_acc
    
def use_method(dist_mat, method, thres = 0.05, k = 4):
    '''
    Choose the method to build adjacency mat
    
    Args:
        dist_mat: matrix; a matrix has been calculated the Euclidean distance
        method: str; kkn or thres
        
    Returns:
        adjacency matrix
    '''
    adj_mat = np.zeros_like(dist_mat)
    
    if method == 'thres':
        # thres will set a threshold for the whole mat, higher →1，lower→0 
        adj_mat = np.where(dist_mat <= thres, 1, 0) 
        
    elif method == 'knn':
        # knn will find the top nearest 5 channel for each 
        for i in range(dist_mat.shape[0]):
            idx = np.argsort(dist_mat[i])[0: k+1]  # sort in row i
            adj_mat[i, idx] = 1  
        adj_mat = np.maximum(adj_mat, adj_mat.T)  
    return adj_mat  
  