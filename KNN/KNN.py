import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.datasets.samples_generator import make_classification
import kdTree_limzh as kd

# X is a vector of x inputs, y is the output. 200 samples
# For each sample, we have 2 features. The outputs are classified into 3 classes.

X, y = make_classification(n_samples = 200, n_features = 2, n_redundant = 0,
                            n_clusters_per_class=1, n_classes = 3)

clf = neighbors.KNeighborsClassifier(n_neighbors=15, weights = 'distance')
clf.fit(X,y)
h = .05 # stepsize in mesh
x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min,x_max,h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# create color maps
cmap_light = ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000','#003300','#0000FF'])
Z = Z.reshape(xx.shape)

plt.figure()
plt.subplot(121)
plt.title('standard-defined')
plt.pcolormesh(xx,yy,Z,cmap = cmap_light)
plt.scatter(X[:,0], X[:,1], c = y, cmap = cmap_bold)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.subplot(122)

clf1 = kd.KNN_Classifier(X, y)
res = clf1.train(K = 15, X = X, y = y, h = h)
res = res.reshape(xx.shape)
plt.title('self-defined')
plt.pcolormesh(xx, yy, res, cmap = cmap_light)
plt.scatter(X[:,0], X[:,1], c = y, cmap = cmap_bold)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.show()