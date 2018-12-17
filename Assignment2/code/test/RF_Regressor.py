# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 19:34:17 2018

@author: chakrabd
"""

from sklearn.ensemble import  RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing.imputation import Imputer

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
    doRFRegression(X_train,y_train,X_test,y_test)
    selectImportantFeatures(X_train,y_train)

rf_mean_scores =[]

def doRFRegression(X_train,y_train,X_test,y_test):
    print('Now doing RF cross validation')
    rf_range = range(10,100,10)
    for i in rf_range:
        rf = RandomForestRegressor(n_estimators = i, random_state = 19)
        scores = cross_val_score(rf, X_train, y_train.ravel(), cv=5,scoring='mean_squared_error')
        print('scores --> ',scores)
        rf_mean_scores.append(scores.mean())
        
    print(rf_mean_scores)
    print('Length of list', len(rf_mean_scores))
    print('Max of list', max(rf_mean_scores))

    plt.plot(rf_range, rf_mean_scores)
    plt.xlabel('Value of n_estimator for RF')
    plt.ylabel('Cross-validated accuracy')

def selectImportantFeatures(X_train,y_train):
    tree_classifier = RandomForestRegressor(n_estimators=100,random_state=0)
    tree_classifier.fit(X_train,y_train.ravel())
    importances = tree_classifier.feature_importances_
    std = np.std([tree.feature_importances_ for tree in tree_classifier.estimators_],
             axis=0)
    indices = importances.argsort()[::-1][:10] #top 10 features in descending order
    print(indices)
    print("Feature ranking:")

    for f in range(indices.size):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(indices.size), importances[indices],
           color="r",yerr=std[indices], align="center")
    plt.xticks(range(indices.size), indices)
    plt.xlim([-1, indices.size])
    plt.show()
    
if __name__== "__main__":
    main()