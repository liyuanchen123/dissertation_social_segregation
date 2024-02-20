import gzip,pickle
import torch
import pandas as pd
import numpy as np
import networkx as nx 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from torch.utils.data import Dataset


class load_data(Dataset):
    def __init__(self, datapath):
        self.x = np.load(datapath)
        # self.x = np.ones((4994,1))
        # self.x = np.eye(4994)
        # self.x = np.sloadtxt('data/{}.txt'.format(dataset), dtype=float)
        # self.y = np.loadtxt('data/{}_label.txt'.format(dataset), dtype=int)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])),\
               torch.from_numpy(np.array(idx))
            #    torch.from_numpy(np.array(self.y[idx])),\
               


def build_graph(datapath):
    datadf = pd.read_pickle(datapath)
    datadf = pd.DataFrame(datadf)

    lsoalist = np.unique(np.array(datadf.fillna('0'))).tolist()
    lsoalist.remove('0')
    voc_size = len(lsoalist)

    # lsoalist = u
    lsoadict = {w:i for i,w in enumerate(lsoalist)}



    pairs = []
    for _, sequence in datadf.iterrows():
        sequence.dropna(inplace=True)
        sequence = sequence.tolist()
        for i in range(len(sequence)-1):
            # for each window
            pairs.append([sequence[i], sequence[i+1]])

    adj = [[0]* len(lsoalist) for _ in range(len(lsoalist))] # shape: 4994 x 4994
    # w = 0
    for i, j in pairs:
        k = lsoadict[i]
        p = lsoadict[j]
        adj[k][p] += 1
        adj[p][k] += 1
        adj[k][k] = 1

    return adj

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.float_power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# sns.heatmap(adj, cmap='RdBu')
def load_graph(path):
    return torch.load(path)

if __name__ == '__main__':
    adj = build_graph(r'data\trajectories\2022_0207_0213.pkl')

    np.save('adj22_unnorm.npy',adj)
    adj = normalize(np.array(adj))
    adj = coo_matrix(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    torch.save(adj, "adj22.pt")