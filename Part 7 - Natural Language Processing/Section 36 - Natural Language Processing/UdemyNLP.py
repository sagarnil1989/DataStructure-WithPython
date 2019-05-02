#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 22:09:28 2019

@author: sagarnildasgupta
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
#import the data
dataset= pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

#cleaning the text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review=review.lower()
    review=review.split()
    ps= PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
    
    #creating bog of word model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X= cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values
    
# splitting the dataset into test set and training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=0)

## feature scaling
#from sklearn.processing import StandardScaler
#sc= StandardScaler()
#X_train=sc.fit_transform(X_train)
#X_test= sc.transform(X_test)

#Fit Naive Bayes to the training set
from sklearn.naive_bayes import GaussianNB
classifier= GaussianNB
classifier.fit(X_train,y_train)

#predicting test set results
y_pred=classifier.predict(X_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

 
    
    