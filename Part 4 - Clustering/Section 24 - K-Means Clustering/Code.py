# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#Applying to dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)
y_means=kmeans.fit_predict(X)

#Visualising the Cluster
plt.close()
plt.scatter(X[y_means==0,0],X[y_means==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(X[y_means==1,0],X[y_means==1,1],s=100,c='blue',label='Cluster 2')
plt.scatter(X[y_means==2,0],X[y_means==2,1],s=100,c='green',label='Cluster 3')
plt.scatter(X[y_means==3,0],X[y_means==3,1],s=100,c='brown',label='Cluster 4')
plt.scatter(X[y_means==4,0],X[y_means==4,1],s=100,c='pink',label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title('Cluster of Clients')
plt.xlabel('Annual Income(k$')
plt.ylabel('Spending score(1-100)')
plt.legend()
plt.show()