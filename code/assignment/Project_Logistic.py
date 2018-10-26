# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 17:14:37 2018

@author: chakrabd
"""

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
from sklearn.linear_model import LogisticRegression  
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import seaborn as sns 
import os as os
from sklearn.metrics import log_loss, confusion_matrix,accuracy_score

print(os.getcwd())
dataset = pd.read_csv('../../dataset/movie_metadata.csv')

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
dataset.head()
#plotting heat map to visualize correlation:
plt.figure(figsize=(18,8),dpi=100,)
plt.subplots(figsize=(18,8))
sns.heatmap(data=dataset.corr(),square=True,vmax=0.8,annot=True)


lb_director = LabelEncoder()
lb_country = LabelEncoder()
lb_language = LabelEncoder()
lb_content_rating = LabelEncoder()
lb_verdict = LabelEncoder()
dataset["director_code"] = lb_director.fit_transform(dataset["director_name"])
dataset["country_code"] = lb_country.fit_transform(dataset["country"])
dataset["lang_code"] = lb_language.fit_transform(dataset["language"])
dataset["content_rating_code"] = lb_content_rating.fit_transform(dataset["content_rating"])

dataset['verdict']=pd.cut(dataset['imdb_score'],bins=[0,7,8,8.5,9,10],labels=["poor","average","good","very good","excellent"],right=False)
dataset['verdict'] = lb_verdict.fit_transform(dataset['verdict'] ) 

dataset.info()
#Setting predictors and target variables
X = dataset.iloc[:, np.r_[1:7,11,13:17]].values
X
y = dataset.iloc[:, 18].values
y
enc = OneHotEncoder(categorical_features=[0])
X = enc.fit_transform(X).toarray()
X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

logistic = LogisticRegression()
logistic.fit(X_train,y_train)
y_pred=logistic.predict(X_test)

logistic.predict_proba(X_test)

conf_matrix=confusion_matrix(y_test,y_pred)
print('The confusion matrxi: ', conf_matrix)

print("Accuracy",accuracy_score(y_test,y_pred)*100)