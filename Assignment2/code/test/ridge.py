# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 02:55:18 2018

@author: chakrabd
"""

from sklearn.linear_model import Lasso,RidgeCV
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split

def main():
    dataset = pd.read_csv('../../../../../googleplaystore_modified.csv')
    dataset.info()
    dataset.drop_duplicates(subset=None, keep='first',inplace=True)
    dataset.isna().sum()
    print(dataset.isnull().sum())
    print(dataset.isnull().any().any())
    dataset.dropna(inplace=True)
    #imp = Imputer(missing_values=np.nan, strategy='mean')
    #dataset['Reviews'] = imp.fit_transform(dataset['Reviews'])
    
    onehotencoder=OneHotEncoder(categorical_features= [0])
    
    encode=LabelEncoder()
    dataset['Category'] = encode.fit_transform(dataset['Category'] )
    #dataset['Reviews'] = encode.fit_transform(dataset['Reviews'] )
    #dataset['Size'] = encode.fit_transform(dataset['Size'] )
    #dataset['Installs'] = encode.fit_transform(dataset['Installs'] )
    #dataset['Type'] = encode.fit_transform(dataset['Type'] )
    #dataset['Price'] = encode.fit_transform(dataset['Price'] )
    dataset['Content Rating'] = encode.fit_transform(dataset['Content Rating'] )
    #dataset['Current Version'] = encode.fit_transform(dataset['Current Version'] )
    dataset['Android Ver'] = encode.fit_transform(dataset['Android Ver'] )
    
    X = dataset.iloc[:, np.r_[1,3:7]].values
    y = dataset.iloc[:, np.r_[2]].values

    X=onehotencoder.fit_transform(X).toarray()
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=19)
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    doRidgeRegressionCV(X_train,y_train,X_test,y_test)
    
rf_mean_scores =[]
rf_std_scores =[]

def doRidgeRegression(X_train,y_train,X_test,y_test):
    print('Now doing SVR cross validation')
    alpha_ridge = [0.001,0.01,0.1,1,10,100]
    for i in alpha_ridge:
        rf = Ridge(alpha=i)
        #rf.score(X_train, y_train, sample_weight=None)
        scores = -cross_val_score(rf, X_train, y_train.ravel(), cv=10,scoring='neg_mean_squared_error')
        print('scores --> ',scores)
        rf_mean_scores.append(scores.mean())
        #rf_std_scores.append(scores.std())
        
    print(rf_mean_scores)
    #print(rf_std_scores)
    print('Length of list', len(rf_mean_scores))
    print('Max of list', max(rf_mean_scores))

    plt.plot(alpha_ridge, rf_mean_scores)
    plt.xlabel('Value of alpha for Ridge')
    plt.ylabel('Cross-validated MSE')
    

def doRidgeRegressionCV(X_train,y_train,X_test,y_test):
    print('Now doing Ridge cross validation')
    alpha_ridge = [0.001,0.01,0.1,1,10,100, 500, 1000]
    rf = RidgeCV(alphas=alpha_ridge,store_cv_values=True).fit(X_train,y_train)
    print(rf.score(X_test,y_test))
    cv_mse = np.mean(rf.cv_values_, axis=0)
    print("alphas: %s" % alpha_ridge)
    print("CV MSE: %s" % cv_mse)
    print("Best alpha using built-in RidgeCV: %f" % rf.alpha_)
    
    plt.plot(np.array(alpha_ridge,dtype="float64"),np.array( cv_mse,dtype="float64")[0])
    plt.xlabel('Value of alpha for Ridge')
    plt.ylabel('Cross-validated MSE')

def doLassoRegression(X_train,y_train,X_test,y_test):
    print('Now doing SVR cross validation')
    alpha_lasso = [0.001,0.01,0.1,1,10]
    for i in alpha_lasso:
        rf = Lasso(alpha=i)
        #rf.score(X_train, y_train, sample_weight=None)
        scores = -cross_val_score(rf, X_train, y_train.ravel(), cv=10,scoring='neg_mean_squared_error')
        print('scores --> ',scores)
        rf_mean_scores.append(scores.mean())
        #rf_std_scores.append(scores.std())
        
    print(rf_mean_scores)
    #print(rf_std_scores)
    print('Length of list', len(rf_mean_scores))
    print('Max of list', max(rf_mean_scores))

    plt.plot(alpha_lasso, rf_mean_scores)
    plt.xlabel('Value of alpha for Ridge')
    plt.ylabel('Cross-validated MSE')
    
if __name__== "__main__":
    main()