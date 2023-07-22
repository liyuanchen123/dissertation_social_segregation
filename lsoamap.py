import geopandas as gpd
import pandas as pd
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

boundary = gpd.read_file(r'C:\UCL\Dissertation\code\data\LSOA_Dec_2021_Boundaries_Full_Clipped_EW_BFC_2022_4005706377815092351\LSOA_2021_EW_BFC_V7.shp')

datadf = pd.read_pickle(r'C:\UCL\Dissertation\code\data\trajectories\2020_0203_0209.pkl')
datadf = pd.DataFrame(datadf)

lsoalist = np.unique(np.array(datadf.fillna('0'))).tolist()
lsoalist.remove('0')
voc_size = len(lsoalist)

# lsoalist = u
lsoadict = {w:i for i,w in enumerate(u)}

clusterlist = torch.load(r'C:\UCL\Dissertation\code\resulttest.pt')
clusterlist = pd.DataFrame(clusterlist)
clusterlist.insert(0, 'lsoacd', lsoalist)
clusterlist.rename(columns={0:'cluster'},inplace=True)
boundary = boundary.merge(clusterlist, left_on='LSOA21CD', right_on='lsoacd', how='right')
######################################
fig, ax = plt.subplots(1, figsize =(16, 8),
                       facecolor ='lightblue')
boundary.plot(ax=ax, column = 'cluster', cmap ='RdBu', legend=True, categorical=True)
# axis for the color bar
# div = make_axes_locatable(ax)
# cax = div.append_axes("right", size ="3 %", pad = 0.05)
  
# # color bar
# vmax = boundary['cluster'].max()
# mappable = plt.cm.ScalarMappable(cmap ='RdBu',
#                                  norm = plt.Normalize(vmin = 0, vmax = vmax))
# cbar = fig.colorbar(mappable, cax)

ax.axis('off')




