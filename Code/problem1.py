"""
 EE 219 Project 4 Problem 1
 Name: Weikun Han
 Date: 3/4/2017
 Reference:
  - https://google.github.io/styleguide/pyguide.html
  - http://scikit-learn.org/stable/
 Description:
  - Clustering
  - Term Frequency-Inverse Document Frequency (TFxIDF) Metric
"""

from __future__ import print_function
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
import numpy as np

# Print information
print(__doc__)
print()

# List categories we want put in dataset
categories = ['comp.graphics',
              'comp.os.ms-windows.misc',
              'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware',
              'rec.autos',
              'rec.motorcycles',
              'rec.sport.baseball',
              'rec.sport.hockey']

# Print information
print("Loading 20 newsgroups dataset for categories...")
print(categories)
print()

# Load dataset
dataset = fetch_20newsgroups(subset = 'all',
                             categories = categories,
                             shuffle = True,
                             random_state = 42)

# Print information
print("-------------------------Processing Finshed 1---------------------------")
print("%d documents" % len(dataset.data))
print("%d categories" % len(dataset.target_names))
print("------------------------------------------------------------------------")
print()

# Print information
print("Transforming the documents into TF-IDF vectors...")
print()

# Transform the documents into TF-IDF vectors
t0 = time()
vectorizer = TfidfVectorizer(max_df = 0.5,
                             max_features = 100000,
                             min_df = 2,
                             stop_words = 'english',
                             use_idf = True)
X = vectorizer.fit_transform(dataset.data)

# Print information
print("-------------------------Processing Finshed 2---------------------------")
print("Transform the documents done in %fs" % (time() - t0))
print("Total samples done: %d, Total features done: %d" % X.shape)
print("------------------------------------------------------------------------")
print()
