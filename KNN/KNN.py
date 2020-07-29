import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.datasets.samples_generator import make_classification

# X is a vector of x inputs, y is the output. 200 samples
# For each sample, we have 2 features. The outputs are classified into 3 classes.

X, y = make_classification(n_samples = 200, n_features = 2, n_redundant = 0,
                            n_clusters_per_class=1, n_classes = 3)

clf = neighbors.KNeighborsClassifier(n_neighbors=1, weights = 'distance')
clf.fit(X,y)
h = .02 # stepsize in mesh
x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
y_min, y_max = X[:,1].min() - 1, X[:,0].max() + 1
xx, yy = np.meshgrid(np.arange(x_min,x_max,h), np.arange(y_min, y_max, h))
print(yy)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# create color maps
cmap_light = ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000','#003300','#0000FF'])
print(Z.shape)
Z = Z.reshape(xx.shape)

####
clf1 = neighbors.RadiusNeighborsClassifier(radius=15.0, weights = 'distance')
clf1.fit(X,y)
Z1 = clf1.predict(np.c_[xx.ravel(), yy.ravel()])
Z1 = Z1.reshape(xx.shape)

plt.figure()
plt.subplot(121)
plt.pcolormesh(xx,yy,Z,cmap = cmap_light)
plt.scatter(X[:,0], X[:,1], c = y, cmap = cmap_bold)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.subplot(122)
plt.pcolormesh(xx,yy,Z1,cmap = cmap_light)
plt.scatter(X[:,0], X[:,1], c = y, cmap = cmap_bold)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.show()