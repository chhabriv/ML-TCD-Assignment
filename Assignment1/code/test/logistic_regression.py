# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 13:35:37 2018

@author: chakrabd
"""

#import matplotlib.pyplot as plt
#from sklearn import datasets, linear_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

data = pd.read_csv('U:/ML/dataset/imdb_1000.csv')
data.shape
data.head()
data.describe()
lb_genre = LabelEncoder()
lb_content = LabelEncoder()
data["genre_code"] = lb_genre.fit_transform(data["genre"])
data["content_code"] = lb_content.fit_transform(data["content_rating"].astype(str))
X=data[['duration','star_rating','content_code']]
y=data[['genre_code']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

logistic = LogisticRegression()
logistic.fit(X_train,y_train)
y_pred=logistic.predict(X_test)
#print(X_test,y_test)
#print('real class is ',y_test,' and predicted class is ', y_pred)
logistic.predict_proba(X_test)
conf_matrix=confusion_matrix(y_test.values,y_pred)
conf_matrix

X=data[['star_rating']]
y=data[['genre_code']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

logistic = LogisticRegression()
logistic.fit(X_train,y_train)
y_pred=logistic.predict(X_test)
#print(y_test.shape)
#print('real class is ',y_test,' and predicted class is ', y_pred)
logistic.predict_proba(X_test)
#conf_matrix = confusion_matrix()
#print(y_test.values)
conf_matrix=confusion_matrix(y_test.values,y_pred)
conf_matrix
#print(classification_report(y_test, y_pred))

X=data[['duration']]
y=data[['genre_code']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

logistic = LogisticRegression()
logistic.fit(X_train,y_train)
y_pred=logistic.predict(X_test)
#print(y_test.shape)
#print('real class is ',y_test,' and predicted class is ', y_pred)
logistic.predict_proba(X_test)
#conf_matrix = confusion_matrix()
#print(y_test.values)
conf_matrix=confusion_matrix(y_test.values,y_pred)
conf_matrix

X=data[['content_code']]
y=data[['genre_code']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

logistic = LogisticRegression()
logistic.fit(X_train,y_train)
y_pred=logistic.predict(X_test)
#print(y_test.shape)
#print('real class is ',y_test,' and predicted class is ', y_pred)
logistic.predict_proba(X_test)
#conf_matrix = confusion_matrix()
#print(y_test.values)
conf_matrix=confusion_matrix(y_test,y_pred)
conf_matrix