# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy import stats
print(os.listdir("//tholospg.itserv.scss.tcd.ie/Pgrad/barmanra/My Documents/ml_play"))

data=pd.read_csv(r"//tholospg.itserv.scss.tcd.ie/Pgrad/barmanra/My Documents/ml_play/movie_metadata.csv")
data.info()
data.head()
data.isna().sum()
# Any results you write to the current directory are saved as output.

#histogram of imdb score

import pylab as pl

imdbScore=[[]]
x=[]

for i in pl.frange(1,9.5,.5):
    imdbScore.append(len(data.imdb_score[(data.imdb_score>=i) & (data.imdb_score<i+.5)]))
    x.append(i)

del(imdbScore[0])

plt.figure(figsize=(15,12))
plt.title("Histogram Of IMDB Score",color="black",size=22)
plt.ylabel("IMDB Score",color="red",size=16)
plt.xlabel('Frequency',color="red",size=16)
plt.barh(x,imdbScore,height=.45 ,color='green')
plt.yticks(x)
plt.show()



#plotting Heat Map

plt.figure(figsize=(18,8),dpi=100,)
plt.subplots(figsize=(18,8))
sns.heatmap(data=data.corr(),square=True,vmax=0.8,annot=True)

from scipy.stats import norm
sns.distplot(a=data['budget'],hist=True,bins=10,fit=norm,color="red")
plt.title("IMDB Movie Rating and Budget")
plt.ylabel("frequency")
mu,sigma=norm.fit(data['budget'])
print("\n mu={:.2f} and sigma={:.2f}\n ".format(mu,sigma))
plt.legend(["normal distribution.($\mu=${:.2f} and $\sigma=${:.2f})".format(mu,sigma)])
plt.show()


#imdb vs budget # scatter plot

plt.figure(figsize=(18,8),dpi=100,)
plt.scatter(x=data['budget'],y=data['imdb_score'],alpha=0.8,color="green")
plt.ylabel("IMDB Score",color="red",size=16)
plt.xlabel("Budget",color="red",size=16)
plt.title("Budget VS IMDB score",color="black",size=22);

#imdb vs no_of_critics reviews

plt.figure(figsize=(18,8),dpi=100,)
plt.scatter(x=data['num_critic_for_reviews'],y=data['imdb_score'],alpha=0.8,color="green")
plt.ylabel("IMDB Score",color="red",size=16)
plt.xlabel("No. Of Critics Reviews",color="red",size=16)
plt.title("Total Critic Reviews VS IMDB score",color="black",size=22);


# IMDB Score V/s Total Facebook Likes

plt.figure(figsize=(18,8),dpi=100,)
plt.scatter(x=data['cast_total_facebook_likes'],y=data['imdb_score'],alpha=0.8,color="green")
plt.ylabel("IMDB Score",color="red",size=16)
plt.xlabel("Cast Total Facebook Likes",color="red",size=16)
plt.title("Cast Facebook Likes VS IMDB score",color="black",size=22);


#gross vs imdb score

plt.figure(figsize=(18,8),dpi=100,)
plt.scatter(x=data['gross'],y=data['imdb_score'],alpha=0.8,color="green")
plt.ylabel("IMDB Score",color="red",size=16)
plt.xlabel("Gross",color="red",size=16)
plt.title("Gross VS IMDB score",color="black",size=22);


#director vs average score


director=list(data.director_name.unique())
df=pd.DataFrame(columns=['director','directorScoreMean','directorImdbScore'])

for i in director:
    tmp=list(data.imdb_score[data.director_name==i])
    if len(tmp)>1:
        df=df.append({'director': i,'directorScoreMean': sum(tmp)/len(tmp),'directorImdbScore': tmp},ignore_index=True)
    
tmp=(df.sort_values(['directorScoreMean'],ascending=False)).head(25)
directorByMeanScore=list(tmp.director)
directorByMeanScore.reverse()
#ScoreByMeanScore=list(tmp.directorImdbScore)
#ScoreByMeanScore.reverse()



plt.figure(figsize=(11,11))
for i in range(len(directorByMeanScore)):
    for j in ScoreByMeanScore[i]:
        plt.scatter(i,j,c=j,vmin=6,vmax=10)

plt.xticks(range(25),directorByMeanScore,rotation=90)
plt.title("Top Director vs Their IMDB Score\n Interm Of Their Average IMDB Score ")
plt.ylabel('Movies IMDB Score')
plt.xlabel("\nDirector's Name")
plt.colorbar(fraction=.04)
plt.show()

