# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('marketing_dataset.csv',sep=';')
dataset.describe()
dataset.isnull().sum()

dataset.head()
dataset.dtypes
dataset_encoded=pd.get_dummies(dataset)

X= dataset_encoded
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

#Choosing 3 cluster
#Applying to dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)
y_means = kmeans.fit_predict(X)
plt.close()
#Visualising the Cluster
from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(X)
pca_2d = pca.transform(X)
import pylab as plt
for i in range(0, pca_2d.shape[0]):
    if y_means[i] == 0:
        c1 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='r',marker='+')
    elif y_means[i] == 1:
        c2 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='g',marker='o')
    elif y_means[i] == 2:
        c3 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='b',marker='*')
plt.legend([c1, c2, c3], ['C1', 'C2','C3'])
plt.title('Marketing dataset with 3 clusters and known outcomes')
plt.show()

#Choosing 4 cluster
#Applying to dataset
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 0)
y_means = kmeans.fit_predict(X)
plt.close()
#Visualising the Cluster
from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(X)
pca_2d = pca.transform(X)
import pylab as plt
for i in range(0, pca_2d.shape[0]):
    if y_means[i] == 0:
        c1 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='r',marker='+')
    elif y_means[i] == 1:
        c2 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='g',marker='o')
    elif y_means[i] == 2:
        c3 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='b',marker='*')
    elif y_means[i] == 3:
        c4 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='y',marker='^')
plt.legend([c1, c2, c3, c4], ['C1', 'C2','C3','C4'])
plt.title('Marketing dataset with 4 clusters and known outcomes')
plt.show()

#Choosing 5 cluster
#Applying to dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)
y_means = kmeans.fit_predict(X)
plt.close()
#Visualising the Cluster
from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(X)
pca_2d = pca.transform(X)
import pylab as plt
for i in range(0, pca_2d.shape[0]):
    if y_means[i] == 0:
        c1 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='r',marker='+')
    elif y_means[i] == 1:
        c2 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='g',marker='o')
    elif y_means[i] == 2:
        c3 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='b',marker='*')
    elif y_means[i] == 3:
        c4 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='y',marker='^')
    elif y_means[i] == 4:
        c5 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='grey',marker='>')
plt.legend([c1, c2, c3, c4, c5], ['C1', 'C2','C3','C4','C5'])
plt.title('Marketing dataset with 5 clusters and known outcomes')
plt.show()

