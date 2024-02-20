import pandas as pd
import numpy as np
import os 
import scipy.sparse as sp
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.float_power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def concat_df(datapath):
    result = pd.DataFrame()
    for filepath in os.listdir(path=datapath):
        # print(i)
        filename = filepath.split('.')[0]
        fieldname = filename.split('_', 2)[-1]
        attdf = pd.read_csv(datapath+filepath)
        attdf[attdf.columns[2]] = attdf[attdf.columns[2]].apply(lambda x: str(x) + fieldname)
        attdf = attdf.pivot(index = 'Lower layer Super Output Areas Code', 
                columns = str(attdf.columns[2]),
                values='Observation')
        
        attdf.reset_index(inplace=True,drop=True)
        result = pd.concat([result, attdf],axis=1)

    result.to_csv('att_data_unnorm.csv',index=False)
    x = np.array(result)
    stdscaler = StandardScaler()
    x = stdscaler.fit_transform(x)

    # x = normalize(x)
    np.save('att_data.npy', x)
    df = pd.DataFrame(x, columns=result.columns)
    df.to_csv('att_data.csv',index=False)
    return x, x.shape[1]



att_data = pd.read_csv('pretrain/att_data.csv')
pca = PCA()
principal_components = pca.fit_transform(att_data)

# Calculate the cumulative sum of variance contribution rates
variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)

# get the number of components when retaining 80% of information of the data
n_components = np.argmax(variance_ratio_cumsum >= 0.95) + 1
print(f'number of components: {n_components}')

# Retain 80% of the original data
pca80 = PCA(n_components=n_components)
principal_components = pca80.fit_transform(att_data)
np.save('pretrain/pca_att_data.npy', principal_components)

# Calculate the average of the weights of each feature in all principal components
feature_importances = np.abs(pca80.components_).mean(axis=0)*100
feature_importances = [round(x,2) for x in feature_importances]

# sorted by importance
feature_importances_sorted = pd.DataFrame({'feature': att_data.columns, 'importance': feature_importances}).sort_values('importance', ascending=False)

print(feature_importances_sorted)
print(f'number of features : {feature_importances_sorted.shape[0]}')



# att_data_unnorm = pd.read_csv(r'pretrain\att_data_unnorm.csv')
