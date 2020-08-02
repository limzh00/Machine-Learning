#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Minzhang Li
# FILE: F:\MyGithubs\Machine-Learning\Supervised-Learning\Nearest-Neighbors\code\KNN_sklearn_test.py
# DATE: 2020/08/02 Sun
# TIME: 14:28:30

# DESCRIPTION: This file offers KNN-sclearn with test


import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from sklearn.datasets import  make_classification
from sklearn import  neighbors, datasets
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime

def main():
    # 1. generate samples. 200 samples for which are of two dims, 3-classes labels. note that dont mess up 'class' with 'cluster'
    X, y = make_classification(n_samples=200, n_features=2 ,n_redundant=0, n_clusters_per_class=1, n_classes=3)
    # -- split test set from train set
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    # 2. generate to-be-classified data (predicted data)
    # -- get the range of to-be-classified point 
    x_min, x_max = x_train[:,0].min() - 1, x_train[:,0].max() + 1
    y_min, y_max = x_train[:,1].min() - 1, x_train[:,1].max() + 1
    # -- mesh 
    h = .01 # step size for mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # -- flatten and pair up
    P = np.c_[xx.ravel(),yy.ravel()] # P is an vector of points 
    # To find the best K we should test the result at test set:
    K = np.arange(1,10) # [1,10)
    test_accuracy = np.zeros(len(K))
    train_accuracy = np.zeros(len(K))
    res = []
    for i, k in enumerate(tqdm(K)):
        # 3. generate classifier
        clf = neighbors.KNeighborsClassifier(n_neighbors=k, weights='distance')
        # 4. fit clf with trained-data
        clf.fit(x_train, y_train)
        # 5. predict P with fitted classifier and get the result Z full of prediected value
        Z = clf.predict(P) # note that it requires P is a vector of 1 dimension
        # push in res list
        res.append(Z)
        # evaluate
        train_accuracy[i] = clf.score(x_train, y_train)
        test_accuracy[i] = clf.score(x_test, y_test)
    # 6. plot the result.
    # -- for scattered points (trained data)
    color_bold = ListedColormap(['#156589', '#199934', '#F9AB3B'])
    # -- for the predicted plane (to-be-classified data)
    color_light = ListedColormap(['#B2EBF2','#DCEDC8','#FFE0B2'])
    # note that the backgroud(color_light should be printed before to avoid overlapping)
    for i in range(len(K)):
        plt.subplot(3,3,i+1)
        plt.pcolormesh(xx, yy, res[i].reshape(xx.shape), cmap = color_light)
        plt.scatter(x_train[:,0], x_train[:,1], c=y_train, cmap=color_bold)
        # add title
        plt.title(f'sklearn-{i+1}NN')
    plt.savefig('../img/fig2.png', format='png')

    # 7. plot the scores
    plt.figure()
    plt.title('K-NN Varying number of neighbors')
    plt.plot(K, train_accuracy, label = 'Training accuracy')
    plt.plot(K, test_accuracy,  label = 'Testing accuracy')
    plt.legend()
    plt.xlabel('Numbero of neighbors')
    plt.ylabel('Accuracy')
    plt.savefig('../img/fig1.png', format='png')
    
if __name__ == '__main__':
    # timing
    start = datetime.now()
    main()
    end = datetime.now()
    print(end - start)
