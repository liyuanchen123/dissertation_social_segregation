from __future__ import print_function, division
import argparse
import random
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from data_graph import load_data, load_graph
from GNN import GNNLayer
from sklearn.metrics import silhouette_score
# from evaluation import eva
from collections import Counter
import sys


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z


class SDCN(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, 
                n_input, n_z, n_clusters, v=1):
        super(SDCN, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z = self.ae(x)
        # x_bar, tra1, tra2, tra3, z = [0,0,0,0,0]

        sigma = 0

        # GCN Module
        h = self.gnn_1(x, adj)
        h = self.gnn_2((1-sigma)*h + sigma*tra1, adj)
        h = self.gnn_3((1-sigma)*h + sigma*tra2, adj)
        h1 = self.gnn_4((1-sigma)*h + sigma*tra3, adj)
        h = self.gnn_5((1-sigma)*h1 + sigma*z, adj, active=False)
        predict = F.softmax(h, dim=1)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z, h1


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

# training
def train_sdcn(dataset):
    model = SDCN(500, 500, 2000, 2000, 500, 500,
                n_input=args.n_input,
                n_z=args.n_z,
                n_clusters=args.n_clusters,
                v=1.0).to(device)
    # print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # KNN Graph
    adj = load_graph(r'C:\UCL\Dissertation\code\adj22.pt')
    adj = adj.cuda()

    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    # y = dataset.y
    with torch.no_grad():
        _, _, _, _, z = model.ae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    # eva(y, y_pred, 'pae')

    for epoch in range(200):
        if epoch % 10 == 0:
        # update_interval
            _, tmp_q, pred, _, gcn_emb = model(data, adj)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)
        
            res1 = tmp_q.cpu().numpy().argmax(1)       #Q
            res2 = pred.data.cpu().numpy().argmax(1)   #Z
            res3 = p.data.cpu().numpy().argmax(1)      #P
            # eva(y, res1, str(epoch) + 'Q')
            # eva(y, res2, str(epoch) + 'Z')
            # eva(y, res3, str(epoch) + 'P')

        x_bar, q, pred, _, gcn_emb = model(data, adj)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)

        loss = 0.3 * kl_loss + 0.3 * ce_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # print(pred.shape)
    # print(gcn_emb.shape)
    score = silhouette_score(gcn_emb.cpu().detach().numpy(), pred.data.cpu().numpy().argmax(1), metric='cosine')
    # score = silhouette_score(data.cpu().numpy(), pred.data.cpu().numpy().argmax(1), metric='euclidean')
    print(score)
    # ############
    # torch.save(pred.data.cpu().numpy().argmax(1), 'result/allatt_result6_gcn.pt')
    # torch.save(y_pred, 'result/ae_kmeans_result6_gcn.pt')
    ###########c

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='acm')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--n_clusters', default=6, type=int)

    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--n_input', default=96, type=int)
    parser.add_argument('--pretrain_path', type=str, default='pkl')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    args.pretrain_path = r'pretrain/ae_pretrain.pkl'
    args.n_input = 96
    # args.pretrain_path = 'data/{}.pkl'.format(args.name)
    dataset = load_data(r'pretrain/att_data.npy')

    # print(args)
    # train_sdcn(dataset)

    # using silhouette score to see which cluster n is the best
    for n in range(2,13):
        args.n_clusters = n
        print(args)
        train_sdcn(dataset)


