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
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn import metrics
from math import sqrt

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

finalColumns=["Pruned %","Random Forest MSE","Random Forest MAE","Random Forest RMSE","Random Forest R2",
                              "Linear MSE","Linear MAE","Linear RMSE","Linear R2"]#,
                              #"SVR MSE","SVR MAE","SVR RMSE","SVR R2"]

results=pd.DataFrame(columns=finalColumns) #Dataframe for each result

prePruneCount=datasetEdit.shape[0]
print('Dataset size before pruning: ',prePruneCount)

 #Standardize the features
scaler=StandardScaler()
datasetEdit[['Reviews','Size','Installs','Price']]=scaler.fit_transform(datasetEdit[['Reviews','Size','Installs','Price']])

#incrementally prune
for prune in range(1,21):
    
    resultList=[]
    #Pruning based on review count < 20
    datasetEdit=datasetEdit.drop(datasetEdit[(datasetEdit['Reviews']<prune)].index).reset_index(drop=True)
    prunedCount=datasetEdit.shape[0]
    resultList.append(round(prunedCount/prePruneCount,2))
    #datasetEdit.info()
    
    #Setting predictors and target variables
    #y=datasetEdit2['Rating']
    #X=datasetEdit2.drop(['Rating','App','Genres'],axis=1)
    X = datasetEdit.iloc[:, np.r_[1,3:11]].values
    y = datasetEdit.iloc[:, 2].values
    
    #split categorical labeled data into columns
    onehotencoder=OneHotEncoder(categorical_features= [0,4,6,7])
    X=onehotencoder.fit_transform(X).toarray()
    

    
    #Split to train, validate, test
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=19)
    #X_train,X_validate,y_train,y_validate=train_test_split(X_train,y_train,test_size=0.15,random_state=19)
    
   
    
    # ============================================================================= 
    # # # Fitting Random Forest Regression to the dataset
    # from sklearn.linear_model import LinearRegression
    # regressor = LinearRegression()
    # regressor.fit(X_train, y_train)
    # 
    # print('RMSE:')
    # print(np.sqrt(metrics.mean_squared_error(y_test, regressor.predict(X_test))))
    # print ('')
    # =============================================================================
    
    """
    # Fitting Random Forest Regression to the dataset
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
    regressor.fit(X_train, y_train)
     
    #prediction
    predict=regressor.predict(X_test)
    print('Regressor Score',regressor.score(X_test,y_test))
    
    print('RMSE:')
    print(np.sqrt(metrics.mean_squared_error(y_test, regressor.predict(X_test))))
    print ('')
    """
    scoring = ['neg_mean_absolute_error','neg_mean_squared_error','r2']
    
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 40, random_state = 19)
    regressor.fit(X_train,y_train)
    scores = cross_validate(regressor, X_train, y_train, cv=10, scoring=scoring)
    resultList.append(round(-scores['test_neg_mean_squared_error'].mean(),2))
    resultList.append(round(-scores['test_neg_mean_absolute_error'].mean(),2))
    resultList.append(round(sqrt(-scores['test_neg_mean_squared_error'].mean()),2))
    resultList.append(round(-scores['test_r2'].mean(),2))
    
    from sklearn.linear_model import LinearRegression,Ridge
    regressor= Ridge(alpha=1)
    regressor.fit(X_train,y_train)
    scores = cross_validate(regressor, X_train, y_train, cv=10, scoring=scoring)
    resultList.append(round(-scores['test_neg_mean_squared_error'].mean(),2))
    resultList.append(round(-scores['test_neg_mean_absolute_error'].mean(),2))
    resultList.append(round(sqrt(-scores['test_neg_mean_squared_error'].mean()),2))
    resultList.append(round(-scores['test_r2'].mean(),2))
    

    #results=results.append(pd.Series(resultList,index=["Random Forest Train","Random Forest Validate","Random Forest Test","Logistic Train","Logistic Validate","Logistic Test","SVC Train","SVC Validate","SVC Test"]),ignore_index=True)
    results=results.append(pd.Series(resultList,index=finalColumns),ignore_index=True)
    
resultList
results.to_csv("Accuracy.csv")
print(results)
postPruneCount=datasetEdit.shape[0]
totalRecordsPruned=(prePruneCount-postPruneCount)
percRecordsPruned=totalRecordsPruned/prePruneCount
print('Dataset size after pruning: ',postPruneCount)
print('Dataset records pruned: ',totalRecordsPruned)
print('% of dataset pruned: ',percRecordsPruned)