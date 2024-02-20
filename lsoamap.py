import geopandas as gpd
import pandas as pd
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.colors as colors
# from osgeo import gdal
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.colors as cor
# import cartopy.io.shapereader as sr
# import cartopy.feature as cfeature
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import cartopy.crs as ccrs
# from matplotlib.path import Path
# from matplotlib.patches import PathPatch
# import shapefile


# def add_north(ax, labelsize=18, loc_x=0.88, loc_y=0.85, width=0.06, height=0.09, pad=0.14):
#     """
#     画一个比例尺带'N'文字注释
#     主要参数如下
#     :param ax: 要画的坐标区域 Axes实例 plt.gca()获取即可
#     :param labelsize: 显示'N'文字的大小
#     :param loc_x: 以文字下部为中心的占整个ax横向比例
#     :param loc_y: 以文字下部为中心的占整个ax纵向比例
#     :param width: 指南针占ax比例宽度
#     :param height: 指南针占ax比例高度
#     :param pad: 文字符号占ax比例间隙
#     :return: None
#     """
#     minx, maxx = ax.get_xlim()
#     miny, maxy = ax.get_ylim()
#     ylen = maxy - miny
#     xlen = maxx - minx
#     left = [minx + xlen*(loc_x - width*.5), miny + ylen*(loc_y - pad)]
#     right = [minx + xlen*(loc_x + width*.5), miny + ylen*(loc_y - pad)]
#     top = [minx + xlen*loc_x, miny + ylen*(loc_y - pad + height)]
#     center = [minx + xlen*loc_x, left[1] + (top[1] - left[1])*.4]
#     triangle = mpatches.Polygon([left, top, right, center], color='k')
#     ax.text(s='N',
#             x=minx + xlen*loc_x,
#             y=miny + ylen*(loc_y - pad + height),
#             fontsize=labelsize,
#             horizontalalignment='center',
#             verticalalignment='bottom')
#     ax.add_patch(triangle)


boundary = gpd.read_file(r'C:\UCL\Dissertation\code\data\LSOA_Dec_2021_Boundaries_Full_Clipped_EW_BFC_2022_4005706377815092351\LSOA_2021_EW_BFC_V7.shp')
# a = pd.read_csv('result/allatt_result6.csv')
# a = torch.tensor(a[['cluster']].values)
# torch.save(a,r'result/allatt_result6.pt')
clusterlist = torch.load(r'word2vec.pt')
clusterlist = pd.DataFrame(clusterlist)
clusterlist.rename(columns={0:'cluster'},inplace=True)
# clusterlist.insert(0, 'lsoacd', lsoalist)

# clmapdict = {
#     0:0,
#     1:1,
#     2:2,
#     3:4,
#     4:3,
#     5:5
# }
# clusterlist['cluster'] = clusterlist['cluster'].map(clmapdict)
# torch.save(clusterlist,r'result/allatt22_result6.pt')

boundary = pd.concat([boundary, clusterlist], axis=1)

# temp = pd.DataFrame(boundary[['LSOA21CD','cluster']])
# temp['cluster'] = temp['cluster'].apply(str)
# temp.to_csv('result/allatt_result6.csv')

######################################
fig, ax = plt.subplots(1, figsize =(16, 8))
ax.set_title('20-21')
cmap = colors.ListedColormap(['#dfd8d1', '#eca24d'])
boundary.plot(ax=ax, column = 'cluster', cmap =cmap, legend=True, categorical=True)

# boundary.to_file(r'result\final\ae_kmeans22_result6.shp')

# axis for the color bar
# div = make_axes_locatable(ax)
# cax = div.append_axes("right", size ="3 %", pad = 0.05)
  
# # color bar
# vmax = boundary['cluster'].max()
# mappable = plt.cm.ScalarMappable(cmap ='RdBu',
#                                  norm = plt.Normalize(vmin = 0, vmax = vmax))
# cbar = fig.colorbar(mappable, cax)

ax.axis('off')




