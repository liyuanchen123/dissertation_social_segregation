# Porcess the POI data
import pandas as pd
import geopandas as gpd
import numpy as np
import os 
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors


att_df = pd.read_csv('att_data.csv')
boundary = gpd.read_file(r'data\LSOA_Dec_2021_Boundaries_Full_Clipped_EW_BFC_2022_4005706377815092351\LSOA_2021_EW_BFC_V7.shp')

clusterlist = torch.load(r'result/allatt_result6.pt')
clusterlist = pd.DataFrame(clusterlist)
# clusterlist.insert(0, 'lsoacd', lsoalist)
clusterlist.rename(columns={0:'cluster'},inplace=True)
boundary = boundary[['LSOA21CD','geometry']].join(clusterlist)
att_df = boundary.join(att_df)

###################
poi_df = gpd.read_file(r'data\Download_london_poi_2015976\poi_4563757\poi_4563757.gpkg')
poi_df = poi_df[['classname','pointx_class', 'geometry']]
poi_att_df = att_df.sjoin(poi_df)
# poi_att_df.to_file(r'data\Download_london_poi_2015976\poi_4563757\poi_london.shp')

poi_dict = {
    '01':'Accomodation eatdrk',
    '02':'Commercial service',
    '03':'Attractions',
    '04':'Sport entertainment',
    '05':'Education health',
    '06':'Public infrastructure',
    '07':'Manufac production',
    '09':'Retail',
    '10': 'Transport'
}

poi_att_df['topclass_code'] = poi_att_df['pointx_class'].apply(lambda x: x[0:2])
poi_att_df['topclass'] = poi_att_df['pointx_class'].apply(lambda x: x[0:2]).map(poi_dict)

# def poi_percent_cal(x):
#     total = len(x)
#     x.groupby('topclass_code')


tempdict = poi_att_df.groupby('LSOA21CD')['topclass'].apply(lambda x: x.value_counts(normalize=True) * 100).to_dict()
index = pd.MultiIndex.from_frame(poi_att_df[['LSOA21CD', 'topclass']])
poi_att_df['poi_percent'] = index.map(tempdict)

poi_percent_df = poi_att_df.pivot_table(index = 'LSOA21CD', columns = ['topclass'], values=['poi_percent'])
poi_percent_df.fillna(0,inplace=True)
poi_percent_df.columns = [s2 for (s1,s2) in poi_percent_df.columns.tolist()]
poi_percent_df.reset_index(inplace=True)
att_poi_final = poi_percent_df.merge(att_df,on='LSOA21CD')

att_poi_final = gpd.GeoDataFrame(att_poi_final)
# att_poi_final.to_file(r'data\Download_london_poi_2015976\poi_4563757\poi_london.shp')
att_poi_final.to_file(r'data\Download_london_poi_2015976\poi_4563757\poi_london.geojson', driver='GeoJSON')

# att_poi_final = gpd.read_file(r'data\Download_london_poi_2015976\poi_4563757\poi_london.shp')

att_poi_data = gpd.read_file(r'data\Download_london_poi_2015976\poi_4563757\poi_london.geojson')


temp = np.array(att_poi_data.iloc[:,1:10])
stdscaler = StandardScaler()
temp = stdscaler.fit_transform(temp)
temp = pd.DataFrame(temp, columns=att_poi_data.iloc[:,1:10].columns)
att_data = pd.read_csv('pretrain/att_data.csv')
att_poi_data = pd.concat([att_data, temp],axis=1)
att_poi_data.to_csv('pretrain/att_poi_data.csv',index=False)
np.save('pretrain/att_poi_data.npy', np.array(att_poi_data))

# # x = normalize(x)
# np.save('att_data.npy', x)
# df = pd.DataFrame(x, columns=result.columns)
#####################################################
boundary = gpd.read_file(r'data\LSOA_Dec_2021_Boundaries_Full_Clipped_EW_BFC_2022_4005706377815092351\LSOA_2021_EW_BFC_V7.shp')

clusterlist = torch.load(r'result/allatt_result6.pt')
clusterlist = pd.DataFrame(clusterlist)
# clusterlist.insert(0, 'lsoacd', lsoalist)
clusterlist.rename(columns={0:'cluster'},inplace=True)
boundary = boundary[['LSOA21CD','geometry']].join(clusterlist)

poi_df = gpd.read_file(r'data\Download_london_poi_2015976\poi_4563757\poi_4563757.gpkg')
poi_df = poi_df[['classname','pointx_class', 'geometry']]
poi_att_df = boundary.sjoin(poi_df)
# poi_att_df.to_file(r'data\Download_london_poi_2015976\poi_4563757\poi_london.shp')

poi_dict = {
    '01':'Accomodation, eating and drinking',
    '02':'Commercial service',
    '03':'Attractions',
    '04':'Sport and entertainment',
    '05':'Education and health',
    '06':'Public infrastructure',
    '07':'Manufacturing production',
    '09':'Retail',
    '10':'Transport'
}

poi_att_df['topclass_code'] = poi_att_df['pointx_class'].apply(lambda x: x[0:2])
poi_att_df['topclass'] = poi_att_df['pointx_class'].apply(lambda x: x[0:2]).map(poi_dict)


tempdict = poi_att_df.groupby('LSOA21CD')['topclass'].apply(lambda x: x.value_counts()).to_dict()
index = pd.MultiIndex.from_frame(poi_att_df[['LSOA21CD', 'topclass']])
poi_att_df['poi_counts'] = index.map(tempdict)

poi_cnt_df = poi_att_df.pivot_table(index = 'LSOA21CD', columns = ['topclass'], values=['poi_counts'])
poi_cnt_df.fillna(0,inplace=True)
poi_cnt_df.columns = [s2 for (s1,s2) in poi_cnt_df.columns.tolist()]
poi_cnt_df.reset_index(inplace=True)
att_poi_final = gpd.GeoDataFrame(poi_cnt_df.merge(boundary,on='LSOA21CD'))

poi_density = att_poi_final.groupby('cluster')[['Accomodation, eating and drinking',
                           'Commercial service',
                           'Attractions',
                           'Sport and entertainment',
                           'Education and health',
                           'Public infrastructure',
                           'Manufacturing production',
                           'Retail',
                           'Transport']].apply(lambda x: x.sum()/att_poi_final['geometry'].area.sum()*1e6)
poi_density.to_csv('result/final/poi_density.csv',index=False)

fig, ax = plt.subplots(figsize=(10, 6))
cmap = sns.color_palette("RdBu_r", as_cmap=True)
bounds = np.array([0,0.5,0.8,1,2,3,5,8,10,15,20,25])
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
sns.heatmap(poi_density, cmap=cmap, norm=norm,ax=ax, annot=True, cbar_kws={'ticks': bounds})
