"""
 EE 219 Project 4 Problem 3 Part 2
 Name: Weikun Han
 Date: 3/6/2017
 Reference:
  - https://google.github.io/styleguide/pyguide.html
  - http://scikit-learn.org/stable/
 Description:
  - Clustering
  - Term Frequency-Inverse Document Frequency (TFxIDF) Metric
  - K-Means Clustering with k = 2
  - Reducing the Dimension with NMF		 
"""

from __future__ import print_function
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.decomposition import KernelPCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from time import time
import numpy as np
import matplotlib.pyplot as plt

# Plot information 
def plot_confusion_matrix(cm, 
                          classes,
                          clusters,
                          title = ''):
    plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    xtick_marks = np.arange(len(clusters))
    ytick_marks = np.arange(len(classes))
    plt.xticks(xtick_marks, clusters, rotation = 45)
    plt.yticks(ytick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

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

# Load dataset and split two groups
dataset = fetch_20newsgroups(subset = 'all', 
                             categories = categories,
                             shuffle = True, 
                             random_state = 42)
size = dataset.target.shape[0]
for i in range(0, size):
    if(dataset.target[i] <= 3):
        dataset.target[i] = 0
    else:
        dataset.target[i] = 1  
labels = dataset.target
class_names = ['Computer technology', 'Recreational activity']
kvalue = 2
cluster_names = []
for i in range(kvalue):
    cluster_names.append("cluster %d" % i)

# Print information
print("Transforming the documents into TF-IDF vectors...")
print()

# Transform the documents into TF-IDF vectors
vectorizer = TfidfVectorizer(max_df = 0.5,
                             max_features = 100000,
                             min_df = 2,
                             stop_words = 'english',
                             use_idf = True)
X = vectorizer.fit_transform(dataset.data)

# Print information
print("Performing dimensionality reduction using NMF without non-linear transformation...")
print()

# Reduce dimension for TF-IDF vectors documents
t0 = time()
svd1 = NMF(n_components = 50, random_state = 2)
normalizer = Normalizer(copy = False)
nmf = make_pipeline(svd1, normalizer)
X_nmf = nmf.fit_transform(X)

# Print information
print("-------------------------Processing Finshed 1---------------------------")
print("Performing dimensionality reductionn done in %fs" % (time() - t0))
print("Total samples done: %d, Total features done: %d" % X_nmf.shape)
print("------------------------------------------------------------------------")
print()

# Print information
print("Performing dimensionality reduction using NMF add non-linear transformation...")
print()

# Reduce dimension for TF-IDF vectors documents
t0 = time()
svd2 = KernelPCA(n_components = 50, kernel='rbf')
normalizer = Normalizer(copy = False)
nmf_nonlinear = make_pipeline(svd2, normalizer)
X_nmf_nonlinear = nmf_nonlinear.fit_transform(X)

# Print information
print("-------------------------Processing Finshed 2---------------------------")
print("Performing dimensionality reductionn done in %fs" % (time() - t0))
print("Total samples done: %d, Total features done: %d" % X_nmf_nonlinear.shape)
print("------------------------------------------------------------------------")
print()

# Print information
print("Clustering sparse data with k-means with k = 2...")
print()

# K-Means clustering with k = kvalue = 2
t0 = time()
km = KMeans(n_clusters = kvalue, init = 'k-means++', max_iter=100, n_init=1, verbose = False)
#km = MiniBatchKMeans(n_clusters=2, init = 'k-means++', n_init=1, init_size = 1000, batch_size = 1000, verbose = False)
km.fit(X_nmf)
original_space_centroids = svd1.inverse_transform(km.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
cm = metrics.confusion_matrix(labels, km.labels_)

# Print information
print("-------------------------Processing Finshed 3---------------------------")
print("Cluster sparse data  done with k-means with k = 2 in %fs" % (time() - t0))
print("This k-means cluster with dimensionality reduction using NMF without non-linear transformation)")
print("Top 10 terms per cluster:")
for i in range(kvalue):
    print("Cluster %d:" % i, end='')
    for j in order_centroids[i, :10]:
        print(' %s' % terms[j], end='')
    print()
print("Confusion matrix:")
print(cm)
print("Homogeneity score: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness score: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("Adjusted rand score: %.3f" % metrics.adjusted_rand_score(labels, km.labels_))
print("Adjusted mutual info score: %0.3f" % metrics.adjusted_mutual_info_score(labels, km.labels_))
print("------------------------------------------------------------------------")
print()

# Plot imformation
plt.figure()
plot_confusion_matrix(cm, 
                      classes = class_names,
                      clusters = cluster_names,
                      title = 'Confusion matrix after NMF without non-linear transformation')

# Print information
print("Clustering sparse data with k-means with k = 2...")
print()

# K-Means clustering with k = kvalue = 2
t0 = time()
km = KMeans(n_clusters = kvalue, init = 'k-means++', max_iter = 100, n_init = 1, verbose = False)
#km = MiniBatchKMeans(n_clusters = kvalue, init = 'k-means++', n_init = 1, init_size = 1000, batch_size = 1000, verbose = False)
km.fit(X_nmf_nonlinear)
original_space_centroids = svd1.inverse_transform(km.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
cm = metrics.confusion_matrix(labels, km.labels_)

# Print information
print("-------------------------Processing Finshed 4---------------------------")
print("Cluster sparse data  done with k-means with k = 2 in %fs" % (time() - t0))
print("This k-means cluster with dimensionality reduction using NMF add non-linear transformation")
print("Top 10 terms per cluster:")
for i in range(kvalue):
    print("Cluster %d:" % i, end='')
    for j in order_centroids[i, :10]:
        print(' %s' % terms[j], end='')
    print()
print("Confusion matrix:")
print(cm)
print("Homogeneity score: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness score: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("Adjusted rand score: %.3f" % metrics.adjusted_rand_score(labels, km.labels_))
print("Adjusted mutual info score: %0.3f" % metrics.adjusted_mutual_info_score(labels, km.labels_))
print("------------------------------------------------------------------------")
print()

# Plot imformation
plt.figure()
plot_confusion_matrix(cm, 
                      classes = class_names,
                      clusters = cluster_names,
                      title = 'Confusion matrix after NMF add non-linear trasformation')
plt.show()


