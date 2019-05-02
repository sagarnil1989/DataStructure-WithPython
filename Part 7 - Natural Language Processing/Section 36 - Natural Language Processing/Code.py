#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 08:20:20 2019

@author: sagarnildasgupta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset= pd.read_csv('Restaurant_Reviews.tsv',sep='\t',quoting=3)

#cleaning the text
import re
import nltk
nltk.download('stopwords') 
from nltk.corpus import stopwords

review = re.sub('[^A-Za-z]',' ',dataset['Review'][0])
review = review.lower()
review= review.split()
review= [word for word in review if not word in set(stopwords.words('english'))]



