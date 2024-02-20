import pandas as pd
import geopandas as gpd
import numpy as np
import os 
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
import seaborn as sns
import matplotlib.colors as colors
import scipy.sparse as sp


class PiecewiseLinearNorm(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, clip=False):
        # call the parent class constructor
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # get the original data range
        x_min, x_max = self.vmin, self.vmax
        # define the range and slope for the desired intervals
        y_min = 0.0  # lower bound of the colorbar
        y_mid1 = 0.9  # middle value of the first interval
        y_mid2 = 0.9  # middle value of the second interval
        y_max = 1.0  # upper bound of the colorbar
        x_mid1 = -100.0  # data value where the first slope changes
        x_mid2 = 100.0  # data value where the second slope changes

        # calculate slopes for the three intervals
        k1 = (y_mid1 - y_min) / (x_mid1 - x_min)
        k2 = (y_mid2 - y_mid1) / (x_mid2 - x_mid1)
        k3 = (y_max - y_mid2) / (x_max - x_mid2)

        # apply the piecewise linear function
        y = np.zeros_like(value)
        mask1 = value < x_mid1
        mask2 = (value >= x_mid1) & (value < x_mid2)
        mask3 = value >= x_mid2

        y[mask1] = y_min + k1 * (value[mask1] - x_min)
        y[mask2] = y_mid1 + k2 * (value[mask2] - x_mid1)
        y[mask3] = y_mid2 + k3 * (value[mask3] - x_mid2)

        # clip values to [0, 1] range if needed
        if clip is None:
            clip = self.clip
        if clip:
            mask = np.ma.getmask(y)
            y = np.ma.array(np.clip(y.filled(self.vmax), 0, 1), mask=mask)

        return y



att_poi_final = gpd.read_file(r'data\Download_london_poi_2015976\poi_4563757\poi_london.geojson')
att_poi_final.groupby('cluster')

#####################################

#### process the attribute excel

att_data = pd.read_csv('pretrain/att_data_unnorm.csv')

clusterlist = torch.load(r'result/allatt22_result6.pt')
clusterlist = pd.DataFrame(clusterlist)
# clusterlist.insert(0, 'lsoacd', lsoalist)
clusterlist.rename(columns={0:'cluster'},inplace=True)
clusterdict = clusterlist.to_dict()['cluster']

att_data = pd.concat([att_data, clusterlist], axis=1)
# att_data.to_csv('pretrain/att_merge.csv',index=False)

# att_data = pd.read_csv('pretrain/att_merge.csv')
att_data['ageunder24'] = att_data.loc[:, ['1age','2age']].sum(axis=1)
att_data['over50'] = att_data.loc[:, ['5age','6age']].sum(axis=1)
att_data['dis2work>30km'] = att_data.loc[:, ['4dis2work','5dis2work']].sum(axis=1)
att_data['asian'] = att_data.loc[:, ['1ethnicity','2ethnicity','3ethnicity','4ethnicity','5ethnicity']].sum(axis=1)
att_data['black'] = att_data.loc[:, ['6ethnicity','7ethnicity','8ethnicity']].sum(axis=1)
att_data['mixed'] = att_data.loc[:, ['9ethnicity','10ethnicity','11ethnicity','12ethnicity']].sum(axis=1)
att_data['white'] = att_data.loc[:, ['13ethnicity','14ethnicity','15ethnicity','16ethnicity','17ethnicity']].sum(axis=1)
att_data['other'] = att_data.loc[:, ['18ethnicity','19ethnicity']].sum(axis=1)

att_data.drop(['1age','2age'],axis=1,inplace=True)
att_data.drop(['5age','6age'],inplace=True,axis=1)
att_data.drop(['4dis2work','5dis2work'],inplace=True,axis=1)
att_data.drop(['1ethnicity','2ethnicity','3ethnicity','4ethnicity','5ethnicity'],inplace=True,axis=1)
att_data.drop(['6ethnicity','7ethnicity','8ethnicity'],inplace=True,axis=1)
att_data.drop(['9ethnicity','10ethnicity','11ethnicity','12ethnicity'],inplace=True,axis=1)
att_data.drop(['13ethnicity','14ethnicity','15ethnicity','16ethnicity','17ethnicity'],inplace=True,axis=1)
att_data.drop(['18ethnicity','19ethnicity'],inplace=True,axis=1)
att_data.drop(['-8dep_edu','-8dep_employ','-8dep_health','-8dep_housing','-8ethnicity'],inplace=True,axis=1)

# att_data.to_csv('pretrain/att_merge_unnorm.csv',index=False)

# att_data = pd.read_csv('pretrain/att_merge.csv')

################## calculate percentage
att_merge_unnorm = pd.read_csv(r'pretrain\att_merge_unnorm.csv')
att_merge_unnorm['population'] = att_merge_unnorm.iloc[:,:4].sum(1)
att_merge_unnorm = att_merge_unnorm.groupby('cluster').sum()
att_merge_unnorm = att_merge_unnorm.drop(['population'],axis=1).apply(lambda x: x/att_merge_unnorm['population']*100)
att_merge_unnorm = att_merge_unnorm.apply(lambda x: round(x, 2))
att_merge_unnorm = att_merge_unnorm.apply(lambda x: (x-x.min())/(x.max()-x.min()))
############

# att

att_data.iloc[:,:-1] = att_data.iloc[:,:-1].apply(lambda x: x-x.min()/x.max()-x.min())
att_data_c = att_data.groupby('cluster').mean()
att_data_c = att_data_c/att_data.drop('cluster',axis=1).mean() * 100
# att_data_c.fillna(0,inplace=True)
# att_data_c = att_data_c.loc[:,list(att_data_c.var()!=0)]
fig, ax = plt.subplots(figsize=(20, 6))
cmap = sns.color_palette("RdBu_r", as_cmap=True)
# cmap = colors.ListedColormap(['#00FF00', '#FF0000', '#000000', '#FFFF00','#00FF00', '#FF0000', '#000000', '#FFFF00','#00FF00', '#FF0000'])
# norm = PiecewiseLinearNorm()
# divnorm = colors.TwoSlopeNorm(vmin=40., vcenter=100, vmax=150)
# bounds = np.array([0,50, 70, 80, 90, 100, 110, 120, 130, 150, 200, 500])
bounds = np.array([0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.95,1])
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
sns.heatmap(att_merge_unnorm, cmap=cmap, norm=norm,ax=ax,  cbar_kws={'ticks': bounds})



# Define your color map
# cmap = sns.color_palette("RdBu_r", as_cmap=True)

# # Define color boundaries
# bounds = np.linspace(0, 1, 11)

# # Create a boundary norm
# norm = colors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N)
# # Set the aspect ratio (height larger than width)
# aspect_ratio = att_merge_unnorm.shape[0] / att_merge_unnorm.shape[1]

# # Create the heatmap with adjusted aspect ratio
# fig, ax = plt.subplots(figsize=(20,6))
# heatmap = ax.imshow(att_merge_unnorm, cmap=cmap, norm=norm, aspect='auto')
# ax.set_yticklabels(att_merge_unnorm.index)
# ax.set_xticklabels(att_merge_unnorm.columns)

# # Add a colorbar with custom ticks
# cbar = plt.colorbar(heatmap, ax=ax, ticks=bounds)

# # Set colorbar labels (optional)
# cbar.set_label("Your Colorbar Label")

# # Show the plot
# plt.show()

########################## interaction adj matrix ################################

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    # mx = mx/rowsum[:,np.newaxis]
    r_inv = np.float_power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

datapath = r'data\trajectories\2020_0203_0209.pkl'
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
    # adj[p][k] += 1
    adj[k][k] = 1

# adj = np.load(r'adj22_unnorm.npy')
# adj = normalize(adj)
# adj = adj.to_dense().numpy()
adj = pd.DataFrame(adj)
# adj.index = [i+1 for i in adj.index] # type: ignore
# adj.columns = [i+1 for i in adj.columns]
adj['cluster'] = adj.index.map(clusterdict)
cwam = adj.groupby('cluster').sum() # sum the rows by cluster labels
cwam = cwam.groupby(cwam.columns.map(clusterdict), axis=1).sum() # sum the columns by cluster labels
cwam = normalize(cwam)
cwam = cwam *100
# cwam = cwam.values # get the numpy array
axcwam=sns.heatmap(cwam, cmap='Reds', annot=True)
axcwam.set_title('2020 Interactions row-percentage matrix (%)')
axcwam.set_xlabel('Destinations')
axcwam.set_ylabel('Origins')
# fig, ax = plt.subplots()
# G = nx.from_numpy_array(cwam,create_using=nx.DiGraph)
# labels = nx.get_edge_attributes(G, "weight")
# pos = nx.spring_layout(G)
# nx.draw(G, pos)
# nx.draw_networkx_edge_labels(G, pos=pos)




###################### maps differences #############

boundary = gpd.read_file(r'data\LSOA_Dec_2021_Boundaries_Full_Clipped_EW_BFC_2022_4005706377815092351\LSOA_2021_EW_BFC_V7.shp')

map_alist = torch.load(r'result/allatt21_result6.pt')
map_alist = pd.DataFrame(map_alist)
# clusterlist.insert(0, 'lsoacd', lsoalist)
map_alist.rename(columns={0:'cluster'},inplace=True)

map_blist = torch.load(r'result/allatt22_result6.pt')
map_blist = pd.DataFrame(map_blist)
# clusterlist.insert(0, 'lsoacd', lsoalist)
map_blist.rename(columns={0:'cluster'},inplace=True)

maplist = map_alist['cluster']-map_blist['cluster']
maplist[maplist != 0] = 1
boundary = pd.concat([boundary, maplist], axis=1)
boundary.to_file(r'result\final\21-22.shp')
