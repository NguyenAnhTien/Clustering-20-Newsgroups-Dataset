"""
 EE 219 Project 4 Problem 4
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
  - Graph-Based K-Means Clustering	 
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
def plot_k_means(X,
                 centroids,
                 labels, 
                 part1,
                 part2,
                 title = ''):
    plt.clf()
    plt.imshow(labels, 
               interpolation = 'nearest',
               extent = (part1.min(), part1.max(), part2.min(), part2.max()),
               cmap = plt.cm.Paired,
               aspect = 'auto', 
               origin = 'lower')
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize = 6)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'x', s = 169, color = 'w', zorder = 10)
    plt.title(title)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

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
svd1 = NMF(n_components = 2, random_state = 2)
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
svd2 = KernelPCA(n_components = 2, kernel='rbf')
normalizer = Normalizer(copy = False)
nmf_nonlinear = make_pipeline(svd2, normalizer)
X_nmf_nonlinear = nmf_nonlinear.fit_transform(X)

# Print information
print("-------------------------Processing Finshed 2---------------------------")
print("Performing dimensionality reductionn done in %fs" % (time() - t0))
print("Total samples done: %d, Total features done: %d" % X_nmf_nonlinear.shape)
print("------------------------------------------------------------------------")
print()

print("Visualizing the results...")
print()

# K-Means clustering with k = kvalue = 2
t0 = time()
km = KMeans(n_clusters = kvalue, init = 'k-means++', max_iter=100, n_init=1, verbose = False)
km.fit(X_nmf)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = 0.02     

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = X_nmf[:, 0].min() - 1, X_nmf[:, 0].max() + 1
y_min, y_max = X_nmf[:, 1].min() - 1, X_nmf[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = km.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)

# Print information
print("-------------------------Processing Finshed 3---------------------------")
print("Visualize the results done in %fs" % (time() - t0))
print("This k-means cluster with dimensionality reduction using NMF without non-linear transformation)")
print("------------------------------------------------------------------------")
print()


# Plot imformation
plt.figure()
plot_k_means(X = X_nmf,
             centroids = km.cluster_centers_,
             labels = Z, 
             part1 = xx,
             part2 = yy,
             title = 'K-means clustering on 20newsgroups dataset \n'
                     'after NMF without non-linear transformation\n'
                     '(Centroids are marked with white cross)')

print("Visualizing the results...")
print()

# K-Means clustering with k = kvalue = 2
t0 = time()
km = KMeans(n_clusters = kvalue, init = 'k-means++', max_iter=100, n_init=1, verbose = False)
km.fit(X_nmf_nonlinear)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = 0.02     

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = X_nmf_nonlinear[:, 0].min() - 1, X_nmf_nonlinear[:, 0].max() + 1
y_min, y_max = X_nmf_nonlinear[:, 1].min() - 1, X_nmf_nonlinear[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = km.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)

# Print information
print("-------------------------Processing Finshed 4---------------------------")
print("Visualize the results done in %fs" % (time() - t0))
print("This k-means cluster with dimensionality reduction using NMF add non-linear transformation")
print("------------------------------------------------------------------------")
print()

# Plot imformation
plt.figure()
plot_k_means(X = X_nmf_nonlinear,
             centroids = km.cluster_centers_,
             labels = Z, 
             part1 = xx,
             part2 = yy,
             title = 'K-means clustering on 20newsgroups dataset \n'
                     'after NMF add non-linear transformation\n'
                     '(Centroids are marked with white cross)')
plt.show()
