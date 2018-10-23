# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 16:12:26 2018

@author: chakrabd
"""

#import matplotlib.pyplot as plt
import numpy as np
#from sklearn import datasets, linear_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  
from sklearn import metrics
data = pd.read_csv('..\..\imdb_1000.csv')
data.shape
data.head()
data.describe()
X=data[['duration']]
y=data[['star_rating']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  
regressor.fit(X_train, y_train)
print(regressor.intercept_)
print(regressor.coef_)
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': [y_test], 'Predicted': [y_pred]})
df
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
