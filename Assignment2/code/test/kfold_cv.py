# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 23:46:09 2018

@author: Aneek,Debrup,Viren
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a sample script file for performing k-fold cross validation for hyper-parameter tuning.
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

def preprocess():
    # Importing the dataset
    dataset = pd.read_csv('../../../Assignment1/dataset/movie_metadata.csv')
    
    dataset.info()
    
    #droping the columns which is not necessary
    dataset.drop(["color","actor_1_facebook_likes","actor_3_facebook_likes",
               "genres","actor_1_name","actor_2_name","movie_title","actor_3_name","facenumber_in_poster",
               "plot_keywords","title_year","movie_imdb_link","actor_2_facebook_likes","aspect_ratio","country","language","director_name"],axis=1,inplace=True)
    
    dataset.isna().sum()
    dataset.info()
    
    #Data imputation
    dataset.replace({"country":np.NaN,
                     #"director_name":np.NaN,
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
    
    datasetEdit=dataset
    
    #encode categorical values
    encode=LabelEncoder()
    datasetEdit['content_rating'] = encode.fit_transform(datasetEdit['content_rating'] ) 
    
    #creating labels based on IMDB score
    datasetEdit['verdict']=pd.cut(datasetEdit['imdb_score'],bins=[0,7,8,8.5,9,10],labels=["poor","average","good","very good","excellent"],right=False)
    datasetEdit['verdict'].value_counts() # Distribution of classes after split
    datasetEdit['verdict'] = encode.fit_transform(datasetEdit['verdict'] ) 
    
    prePruneCount=datasetEdit.shape[0]
    print('Dataset size before pruning: ',prePruneCount)
    return datasetEdit

svc_scores = []

def doSVMClassification(X_train,y_train,X_test,y_test):
    print('Now doing cross validation')
    C_param_range = [0.001, 0.01, 0.1,1, 10, 25, 50, 100, 1000]
    for i in C_param_range:
        svc = SVC(C=i)
        scores = cross_val_score(svc, X_train, y_train, cv=2, scoring='accuracy')
        svc_scores.append(scores.mean())
        
    print(svc_scores)
    print('Length of list', len(svc_scores))
    print('Max of list', max(svc_scores))

    plt.plot(C_param_range, svc_scores)
    plt.xlabel('Value of C for SVC')
    plt.ylabel('Cross-validated accuracy')


rf_scores =[]

def doRFClassification(X_train,y_train,X_test,y_test):
    print('Now doing RF cross validation')
    rf_range = range(10,200,10)
    for i in rf_range:
        rf = RandomForestClassifier(n_estimators = i, random_state = 19)
        scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
        rf_scores.append(scores.mean())
        
    print(rf_scores)
    print('Length of list', len(rf_scores))
    print('Max of list', max(rf_scores))

    plt.plot(rf_range, rf_scores)
    plt.xlabel('Value of n_estimator for RF')
    plt.ylabel('Cross-validated accuracy')


def main():
    datasetEdit = preprocess()
    
    #Setting predictors and target variables
    X = datasetEdit.iloc[:, np.r_[0:9,10]].values
    y = datasetEdit.iloc[:, -1].values
    
    #split categorical labeled data into columns
    onehotencoder=OneHotEncoder(categorical_features= [0])
    X=onehotencoder.fit_transform(X).toarray()
    
    #Split to train, validate, test   
    
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=19)
    #X_train,X_validate,y_train,y_validate=train_test_split(X_train,y_train,test_size=0.15,random_state=19)
    
    #Normalising the features
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    doRFClassification(X_train,y_train,X_test,y_test)
        

if __name__== "__main__":
    main()