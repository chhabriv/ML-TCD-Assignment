# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Importing the dataset
dataset = pd.read_csv('googleplaystore.csv')

dataset.info()

#check NA's in data
dataset.isna().sum()

columns = dataset.columns
percent_missing = round(dataset.isna().sum() * 100 / len(dataset) ,2)
missing_value_df = pd.DataFrame({'column_name': columns,
                                 'percent_missing': percent_missing})
missing_value_df

#Check missingness in data
plt.figure(figsize=(14,10),dpi=100,)
plt.subplots(figsize=(14,10))
sns.heatmap(dataset.isna(), cbar=False)

#droping the columns which is not necessary
dataset.drop(["Current Ver","Last Updated"],axis=1,inplace=True)

dataset=dataset.dropna(subset=['Rating','Type','Content Rating','Android Ver','Category','Genres'])

dataset.replace({'Size':"Varies with device"},value=-1,inplace=True)
dataset['Reviews']=pd.to_numeric(dataset['Reviews'])
dataset['Price']=dataset['Price'].replace('[\$,]', '', regex=True).astype(float)
dataset['Size']=pd.to_numeric(dataset['Size'])
dataset['Installs']=pd.to_numeric(dataset['Installs'])


dataset.info()
datasetEdit=dataset

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
encode=LabelEncoder()
datasetEdit['App'] = encode.fit_transform(datasetEdit['App'] ) 
datasetEdit['Category'] = encode.fit_transform(datasetEdit['Category'] ) 
datasetEdit['Reviews'] = encode.fit_transform(datasetEdit['Reviews'] ) 
datasetEdit['Type'] = encode.fit_transform(datasetEdit['Type'] ) 
datasetEdit['Content Rating'] = encode.fit_transform(datasetEdit['Content Rating'] ) 
datasetEdit['Genres'] = encode.fit_transform(datasetEdit['Genres'] ) 
datasetEdit['Android Ver'] = encode.fit_transform(datasetEdit['Android Ver'] )

#plotting heat map to visualize correlation:
plt.figure(figsize=(18,8),dpi=100,)
plt.subplots(figsize=(18,8))
sns.heatmap(data=datasetEdit.corr(),square=True,vmax=0.8,annot=True)