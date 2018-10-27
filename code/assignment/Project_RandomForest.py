# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 14:59:41 2018

@author: chhabriv
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
from sklearn import metrics
from sklearn.metrics import accuracy_score


# Importing the dataset
dataset = pd.read_csv('movie_metadata.csv')

dataset.info()

#droping the columns which is not necessary
dataset.drop(["color","actor_1_facebook_likes","actor_3_facebook_likes",
           "genres","actor_1_name","actor_2_name","movie_title","actor_3_name","facenumber_in_poster",
           "plot_keywords","title_year","movie_imdb_link","actor_2_facebook_likes","aspect_ratio"],axis=1,inplace=True)

dataset.isna().sum()
dataset.info()

#Data imputation
dataset.replace({"country":np.NaN,
                 "director_name":np.NaN,
              "language":np.NaN,
             "content_rating":np.NaN},value="Missing",inplace=True)

dataset['duration']=dataset['duration'].fillna(value=dataset['duration'].mean())
dataset['num_user_for_reviews']=dataset['num_user_for_reviews'].fillna(value=dataset['num_user_for_reviews'].mean())
dataset['budget']=dataset['budget'].fillna(value=dataset['budget'].mean())
dataset['gross']=dataset['gross'].fillna(value=dataset['gross'].mean())
dataset['director_facebook_likes']=dataset['director_facebook_likes'].fillna(value=0)
dataset['num_critic_for_reviews']=dataset['num_critic_for_reviews'].fillna(value=0)

#remove duplicates
dataset.drop_duplicates(subset=None, keep='first',inplace=True)

dataset.isna().sum()
dataset.info()

#plotting heat map to visualize correlation:
plt.figure(figsize=(18,8),dpi=100,)
plt.subplots(figsize=(18,8))
sns.heatmap(data=dataset.corr(),square=True,vmax=0.8,annot=True)

datasetEdit=dataset

#encode categorical values
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
encode=LabelEncoder()
datasetEdit['director_name'] = encode.fit_transform(datasetEdit['director_name'] ) 
datasetEdit['language'] = encode.fit_transform(datasetEdit['language'] ) 
datasetEdit['country'] = encode.fit_transform(datasetEdit['country'] ) 
datasetEdit['content_rating'] = encode.fit_transform(datasetEdit['content_rating'] ) 

datasetEdit['verdict']=pd.cut(datasetEdit['imdb_score'],bins=[0,7,8,8.5,9,10],labels=["poor","average","good","very good","excellent"],right=False)
datasetEdit['verdict'] = encode.fit_transform(datasetEdit['verdict'] ) 

#incrementally prune
for prune in range(1,11):

    #Pruning based on review count < 10
    datasetEdit=datasetEdit.drop(datasetEdit[(datasetEdit['num_user_for_reviews']<prune)].index).reset_index(drop=True)
    datasetEdit.info()
    
    #Setting predictors and target variables
    X = datasetEdit.iloc[:, np.r_[0:12,13]].values
    y = datasetEdit.iloc[:, 14].values
    
    #split categorical labeled data into columns
    onehotencoder=OneHotEncoder(categorical_features= [0])
    X=onehotencoder.fit_transform(X).toarray()
    
    #Split to train and test
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=19)
    
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
    from sklearn.ensemble import RandomForestClassifier
    regressor = RandomForestClassifier(n_estimators = 10, random_state = 0)
    regressor.fit(X_train, y_train)
     
    #prediction
    predict=regressor.predict(X_test)
    print("Accuracy Random Forest",accuracy_score(y_test,predict)*100)
    
    from sklearn.linear_model import LogisticRegression  
    logistic = LogisticRegression()
    logistic.fit(X_train,y_train)
    y_pred=logistic.predict(X_test)
    print("Accuracy Logistic",accuracy_score(y_test,y_pred)*100)
    
    
    from sklearn.svm import SVC 
    svc = SVC()
    svc.fit(X_train,y_train)
    y_pred1=svc.predict(X_test)
    print("Accuracy svc",accuracy_score(y_test,y_pred1)*100)