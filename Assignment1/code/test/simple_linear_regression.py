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
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt

#Loading the dataset using pandas
data = pd.read_csv('../../dataset/imdb_1000.csv')
data.shape
data.head()
data.describe()
#Category values need to be changed to numerical codes for applying any ML algorithm.
lb_make = LabelEncoder()
data["genre_code"] = lb_make.fit_transform(data["genre"])
data[["genre", "genre_code"]].head(11)
#Initial testing includes one predictor (X) that is 'duration' and our target is 'star_rating'
X=data[['duration']]
y=data[['star_rating']]
#Using the sklearn in-built train-test splitting method to divide the data. Random state '0' or any number indicates it will always split the data into outputs that would be same,
#if it is run multiple times.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#initialising the Linear Regression model constructor
regressor = LinearRegression()  
#Train the model with the input data available
regressor.fit(X_train, y_train)
#plotting the regressor with training dataset
plt.scatter(X_train, y_train,color='g')
plt.plot(X_train, regressor.predict(X_train),color='k')
plt.title('Rating vs Duration (Training set)')
plt.xlabel('Duration')
plt.ylabel('Rating')
plt.show()
#plotting the regressor with test dataset
plt.scatter(X_test, y_test,color='g')
plt.plot(X_train, regressor.predict(X_train),color='k')
plt.title('Rating vs Duration (Training set)')
plt.xlabel('Duration')
plt.ylabel('Rating')
plt.show()

#printing various features of the model to know the quality of the model
print(regressor.intercept_)
print(regressor.coef_)
#predicting using the test dataset
y_pred = regressor.predict(X_test)
#metric to find the performance of the model
df = pd.DataFrame({'Actual': [y_test], 'Predicted': [y_pred]})
df
#finding standard errors in the model
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
