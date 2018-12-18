# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 23:24:50 2018

@author: chakrabd
"""


from sklearn.svm import SVR
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
    doSVRegression(X_train,y_train,X_test,y_test)

rf_mean_scores =[]
rf_std_scores =[]

def doSVRegression(X_train,y_train,X_test,y_test):
    print('Now doing SVR cross validation')
    C_param_range = [0.001, 0.01, 0.1,1, 10, 25, 50, 100, 1000]
    sv_kernels = ['rbf']
    for ker in sv_kernels:
        for i in C_param_range:
            rf = SVR(C = i,kernel = ker)
            #rf.score(X_train, y_train, sample_weight=None)
            scores = cross_val_score(rf, X_train, y_train.ravel(), cv=10,scoring='neg_mean_squared_error')
            print('scores --> ',scores)
            rf_mean_scores.append(scores.mean())
            #rf_std_scores.append(scores.std())
            
        print(rf_mean_scores)
        #print(rf_std_scores)
        print('Length of list', len(rf_mean_scores))
        print('Max of list', max(rf_mean_scores))
    
        plt.plot(C_param_range, rf_mean_scores)
        plt.xlabel('Value of n_estimator for RF')
        plt.ylabel('Cross-validated accuracy')


    
if __name__== "__main__":
    main()