#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 15:49:20 2019

@author: labmatematicas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from pandas import datetime
from sklearn.cluster import KMeans

train = pd.read_csv("/home/labmatematicas/Escritorio/rossman/train.csv",parse_dates = True, low_memory = False,index_col = 'Date')
store = pd.read_csv("/home/labmatematicas/Escritorio/rossman/store.csv")

train['Year'] = train.index.year
train['Month'] = train.index.month
train['Day'] = train.index.day
train['WeekOfYear'] = train.index.weekofyear

train['SalePerCustomer'] = train['Sales']/train['Customers'] #WHY???
train['SalePerCustomer'].describe()

plt.hist(train['SalePerCustomer'][train['Customers']!=0],bins = 50)

###There is a day in which some store is open and there is not sales?
##closed stores
train[(train.Open == 0) & (train.Sales == 0)].head()
prop_zeros = len(train[(train.Open == 0) & (train.Sales == 0)]['Year'])/len(train['Year'])
##open stores with zero sales
zero_sales = train[(train.Open != 0) & (train.Sales == 0)]
len(zero_sales['Year'])

##avoiding zero Sales
train = train[(train['Open']!=0) & (train['Sales']!=0)]

##missing values?
store.isnull().sum()

# fill NaN with a median value (skewed distribuion)
store['CompetitionDistance'].fillna(store['CompetitionDistance'].median(), inplace = True)

# no promo = no information about the promo?
pr2_N = store[pd.isnull(store.Promo2SinceWeek)]
pr2_N[pr2_N.Promo2 != 0].shape
pr2_N[pr2_N.Promo2 != 0].shape

# replace NA's by 0
store.fillna(0, inplace = True)

##merge of data
train_store = pd.merge(train, store, how = 'inner', on = 'Store')

######Store Types
train_store.groupby('StoreType')['Sales'].describe() ##WHY IS NOT THIS ENOUGH INFORMATION???

train_store.groupby('StoreType')['Customers', 'Sales'].sum()

# sales trends
sns.factorplot(data = train_store, x = 'Month', y = "Sales", 
               col = 'StoreType', # per store type in cols
               palette = 'plasma',
               hue = 'StoreType',
               row = 'Promo') # per promo in the store in rows
               
# Day of week vs sales
sns.factorplot(data = train_store, x = 'Month', y = "Sales", 
               col = 'DayOfWeek', # per store type in cols
               palette = 'plasma',
               hue = 'StoreType',
               row = 'StoreType') # per store type in rows
                
# competition open time (in months)
train_store['CompetitionOpen'] = 12 * (train_store.Year - train_store.CompetitionOpenSinceYear) + \
        (train_store.Month - train_store.CompetitionOpenSinceMonth)
    
# Promo open time
train_store['PromoOpen'] = 12 * (train_store.Year - train_store.Promo2SinceYear) + \
        (train_store.WeekOfYear - train_store.Promo2SinceWeek) / 4.0

# replace NA's by 0
train_store.fillna(0, inplace = True)

# average PromoOpen time and CompetitionOpen time per store type
train_store.loc[:, ['StoreType', 'Sales', 'Customers', 'PromoOpen', 'CompetitionOpen']].groupby('StoreType').mean()     

# Compute the correlation matrix 
# exclude 'Open' variable
corr_all = train_store.drop('Open', axis = 1).corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr_all, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize = (11, 9))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_all, mask = mask,
            square = True, linewidths = .5, ax = ax, cmap = "BuPu")      
plt.show()


# sale per customer trends
sns.factorplot(data = train_store, x = 'DayOfWeek', y = "Sales", 
               col = 'Promo', 
               row = 'Promo2',
               hue = 'Promo2',
               palette = 'RdPu') 

##############################
##############################
##############################
sns.factorplot(data = train_store, x = 'Month', y = "Sales", 
               col = 'Assortment', # per store type in cols
               palette = 'plasma',
               hue = 'Assortment')

sns.factorplot(data = train_store, x = 'Month', y = "Sales", 
               col = 'StoreType', # per store type in cols
               palette = 'plasma',
               hue = 'StoreType')

sns.factorplot(data = train_store, x = 'Month', y = "Sales", 
               col = 'Assortment', # per store type in cols
               palette = 'plasma',
               hue = 'Assortment',row = 'Promo')


sns.factorplot(data = train_store, x = 'DayOfWeek', y = "Sales", 
               col = 'Assortment', # per store type in cols
               palette = 'plasma',
               hue = 'Assortment')

sns.factorplot(data = train_store, x = 'DayOfWeek', y = "Sales", 
               col = 'StoreType', # per store type in cols
               palette = 'plasma',
               hue = 'StoreType')

sns.factorplot(data = train_store, x = 'DayOfWeek', y = "Sales", 
               col = 'Month', # per store type in cols
               palette = 'plasma',
               hue = 'Month')

#####Clusters of months based in plots x = "DayOfWeek",y = "Sales"
m_clust1 =[1,2,3,4]
m_clust2 = [5,6,7,8,9,10]
m_clust3 = [11]
m_clust4 =[12]

sns.factorplot(data = train_store, x = 'DayOfWeek', y = "Sales", 
               col = 'WeekOfYear', # per store type in cols
               palette = 'plasma',
               hue = 'WeekOfYear')
#############################################
sns.factorplot(data = train_store, x = 'Month', y = "Sales", 
               col = 'Assortment', # per store type in cols
               palette = 'plasma',
               hue = 'StoreType',row = 'StoreType')

sns.factorplot(data = train_store, x = 'DayOfWeek', y = "Sales", 
               col = 'Assortment', # per store type in cols
               palette = 'plasma',
               hue = 'StoreType',row = 'StoreType')

##############################
##############################
M_weeks = np.zeros([7,52])

M_weeks = [[(train_store[(train_store['WeekOfYear'] == j) & (train_store['DayOfWeek'] == i)]['Sales']).mean() for i in range(1,8)] for j in range(1,53)]

kmeans = KMeans(n_clusters=3)

kmeans.fit(M_weeks)
centros = kmeans.cluster_centers_

clusters = kmeans.fit_predict(M_weeks)

num0 = [i+1 for i in range(0,52) if (clusters[i] == 0)]
num1 = [i+1 for i in range(0,52) if (clusters[i] == 1)]
num2 = [i+1 for i in range(0,52) if (clusters[i] == 2)]

train_store['WeekOfYear'].index(num0)

train_store['clusterWeek'] = 0*train_store['Sales']

train_store['clusterWeek'][sum([train_store['WeekOfYear']== i for i in num0])==1] = 0
train_store['clusterWeek'][sum([train_store['WeekOfYear']== i for i in num1])==1] = 1
train_store['clusterWeek'][sum([train_store['WeekOfYear']== i for i in num2])==1] = 2

train_store[train_store['clusterWeek'] == 0]


corr0 = train_store[train_store['clusterWeek'] == 0].drop('Open', axis = 1).corr()
corr1 = train_store[train_store['clusterWeek'] == 1].drop('Open', axis = 1).corr()
corr2 = train_store[train_store['clusterWeek'] == 2].drop('Open', axis = 1).corr()
# Generate a mask for the upper triangle
mask0 = np.zeros_like(corr0, dtype = np.bool)
mask0[np.triu_indices_from(mask0)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize = (11, 9))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr0, mask = mask,
            square = True, linewidths = .5, ax = ax, cmap = "BuPu")      


mask1 = np.zeros_like(corr1, dtype = np.bool)
mask1[np.triu_indices_from(mask1)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize = (11, 9))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr1, mask = mask,
            square = True, linewidths = .5, ax = ax, cmap = "BuPu")      


mask2 = np.zeros_like(corr2, dtype = np.bool)
mask2[np.triu_indices_from(mask2)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize = (11, 9))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr2, mask = mask,
            square = True, linewidths = .5, ax = ax, cmap = "BuPu")      

plt.show()

corr0['Promo2']
corr1['Promo2']
corr2['Promo2']
corr_all['Promo2']

#########################
#########################

train_store_cluster0=train_store['clusterWeek']==0

cluster0=train_store[train_store_cluster0]

cluster0.head()


sns.factorplot(data = cluster0, x = 'DayOfWeek', y = "Sales", 
               col = 'Assortment',# per store type in cols
               row  = 'StoreType',
               palette = 'plasma',
               hue = 'Assortment')


train_store_cluster1=train_store['clusterWeek']==1

cluster1=train_store[train_store_cluster1]

cluster1.head()


sns.factorplot(data = cluster1, x = 'DayOfWeek', y = "Sales", 
               col = 'Assortment',# per store type in cols
               row  = 'StoreType',
               palette = 'plasma',
               hue = 'Assortment')


train_store_cluster2=train_store['clusterWeek']==2

cluster2=train_store[train_store_cluster2]

cluster2.head()


sns.factorplot(data = cluster2, x = 'DayOfWeek', y = "Sales", 
               col = 'Assortment',# per store type in cols
               row  = 'StoreType',
               palette = 'plasma',
               hue = 'Assortment')


##########################
##########################


train_store_2013=train_store['Year']==2013

TS_Yeat2013=train_store[train_store_2013]

l=sns.factorplot(data = TS_Yeat2013, x = 'DayOfWeek', y = "Sales", 
               col = 'Assortment',# per store type in cols
               row  = 'StoreType',
               palette = 'plasma',
               hue = 'Assortment')
print(train_store['Year'])
##### Segmentation Assortment- Store type-Perrito #############
####################################################################
## Segmentation week perrito: para cada cluster generado po WeekOf Year
##generamos tres clusters nuevos. Las variables de los tres nuevos clusters son ['Assortment','StoreType']
###################################################################
ass_Type=train_store['Assortment'].unique()
sto_Type=train_store['StoreType'].unique() 
#couples=[[i,j] for i in ass_Type  for j in sto_Type]
train_store0 = train_store[train_store['clusterWeek'] == 0]
M_asso0 = [[(train_store0[(train_store0['Assortment'] == i) & (train_store0['StoreType'] == k) & (train_store0['DayOfWeek'] == j)]['Sales']).mean() for j in range(1,8)] for i in ass_Type for k in sto_Type]
 
M_asso0 = np.nan_to_num(np.asarray(M_asso0))

row_zeros = np.where((M_asso0==0).all(axis=1)==True)

M_asso = np.delete(M_asso0,row_zeros,axis = 0)

#######Elegimos el # de clusters como a Hernan le dio la puta gana: necesitamos una manera metódica 
#######de fijar ese numero
kmeans_store = KMeans(n_clusters=3)

kmeans_store.fit(M_asso)
centros_store = kmeans_store.cluster_centers_

clusters_store = kmeans.fit_predict(M_asso)

num0 = [i+1 for i in range(0,len(M_asso)) if (clusters_store[i] == 0)]
num1 = [i+1 for i in range(0,len(M_asso)) if (clusters_store[i] == 1)]
num2 = [i+1 for i in range(0,len(M_asso)) if (clusters_store[i] == 2)]

