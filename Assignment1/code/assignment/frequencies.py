# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 17:39:33 2018

@author: barmanra
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy import stats
from sklearn.linear_model.logistic import _logistic_loss

print(os.listdir("//tholospg.itserv.scss.tcd.ie/Pgrad/barmanra/My Documents/ml_play"))

data=pd.read_csv(r"//tholospg.itserv.scss.tcd.ie/Pgrad/barmanra/My Documents/ml_play/movie_metadata.csv")
data.info()
data.head()
data.isna().sum()
plt.data
# Any results you write to the current directory are saved as output.

#histogram of imdb score


plt.title("Histogram Of IMDB Score",color="black",size=18)
plt.xlabel("IMDB Score",color="red",size=16)
plt.ylabel('Frequency',color="red",size=16)
data['imdb_score'].hist()