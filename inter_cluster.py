from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# Only using interactons to do the cluster, but it doesn't work for some reasons.

class skipgram(nn.Module):
    def __init__(self, worddict, voc_size, emb_dim=10 ) -> None:
        super(skipgram, self).__init__()
        self.worddict = worddict
        self.W = nn.Embedding(voc_size, emb_dim)      
        self.W.weight.requires_grad = True
        # self.W1 = nn.Linear(voc_size, emb_dim, bias=False)
        self.W2 = nn.Linear(emb_dim, voc_size, bias=False)
        self.softmax = nn.Softmax()

    def forward(self, X):
        # print(X, self.embedding.weight)
        # embeddings  = self.W(X)
        # hidden_layer = self.W1(X)
        hidden_layer = self.W(X)
        output_layer = self.W2(hidden_layer)
        # f = self.softmax(output_layer)
        return output_layer
    
# model = skipgram(worddict, voc_size=voc_size, emb_dim=30)
model = torch.load('word2vec.pt')
layers=[x.data for x in model.parameters()]
features = layers[1].cpu().numpy()
kmeans = KMeans(n_clusters=6, n_init=20)
y_pred = kmeans.fit_predict(features)


boundary = gpd.read_file(r'C:\UCL\Dissertation\code\data\LSOA_Dec_2021_Boundaries_Full_Clipped_EW_BFC_2022_4005706377815092351\LSOA_2021_EW_BFC_V7.shp')

clusterlist = pd.DataFrame(y_pred)
# clusterlist.insert(0, 'lsoacd', lsoalist)
clusterlist.rename(columns={0:'cluster'},inplace=True)
boundary = pd.concat([boundary, clusterlist], axis=1)

fig, ax = plt.subplots(1, figsize =(16, 8))
boundary.plot(ax=ax, column = 'cluster', cmap ='RdBu', legend=True, categorical=True)
boundary.to_file(r'result\final\flowmap.shp')