# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 15:14:49 2021

@author: jahna
"""

#importing all the required modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
pd.options.mode.chained_assignment = None


# defining a function which calculates the percentage accuracy
def percentage_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)*100


# importing the data 
df1 = pd.read_csv(r"C:\Users\jahna\Desktop\Academics\Sem-5\PR\assg1\nls_data\class1.txt", header=None, names=["x","y"])
df2 = pd.read_csv(r"C:\Users\jahna\Desktop\Academics\Sem-5\PR\assg1\nls_data\class2.txt", header=None, names=["x","y"])
train = df1.append(df2)



labels1 = [0]*len(df1)
labels2 = [1]*len(df2)
labels = labels1 + labels2
test = np.array(labels)


test=pd.DataFrame(test,columns=["class"])
X_train, X_test, y_train, y_test = train_test_split(train,test, test_size=0.30, random_state=42)
x = X_train["x"]
y = X_train["y"]
df = X_train


colmap = {1: 'r', 2: 'b',3:'g',4:'y'}
np.random.seed(200)
k = 2


# storing random x,y values in a dictionary as centroids
centroids={}
for i in range(k):
    centroids[i+1]=[np.random.randint(min(x),max(x)), np.random.randint(min(y),max(y))]
  
    
# function to assign data points to 'k' clusters
def assignment(df, centroids):
    
    # evaluating the distance of data points from each cluster
    
    for i in centroids.keys():
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                + (df['y'] - centroids[i][1]) ** 2
            )
        )
    

    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    
   
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    # this will store the minimum distance between the data point and clustser center
    # and also from which cluster it is nearer to as "distance_from_{clustercentre}"
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    
    # this will map the cluster centre to its assigned colour
    df['color'] = df['closest'].map(lambda x: colmap[x])
    return df

df = assignment(df, centroids)


# function for updating the centroids as mean of the data points in the cluster
def update(k):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return k



while True:
    # Make a deep copy, including a copy of the data and the indices
    closest_centroids = df['closest'].copy(deep=True)
    
    #update the cluster centres
    centroids = update(centroids)
    
    # assign the data points to different clusters
    #calling kmeans with updated centroids
    df = assignment(df, centroids)
    
    # if the previously assigned cluster and the cluster which is assigned now
    # are equal for all the data points 
    # then break the loop
    # else continue updating the cluster centres
    if closest_centroids.equals(df['closest']):
        break

    # plotting the data points
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.6)
    for i in centroids.keys():
        plt.scatter(*centroids[i], color=colmap[i])
    plt.xlim(-15, 20)
    plt.ylim(-30, 50)
    plt.show()



count0 = 0
count1 = 0
#print(df.head())
#print(y_train)


for i,j in zip(df["closest"],y_train):
  if(i == 1 and j == 0):
   count0 = count0 + 1
  if(i == 1 and j == 1):
    count1 = count1 + 1

df1 = assignment(X_test, centroids)
df1['closest']-=1
#print(df1)

y_pred = (df1["closest"]).to_numpy()
y_true = (y_test).to_numpy()
print(percentage_accuracy(y_test, y_pred))

