# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

from sklearn.cluster import KMeans
for i in range(1,11):
    kmeans=KMeans(n_cluster=i, init='k-means++',max_tier=300,n_init= 10, random_state= 0)