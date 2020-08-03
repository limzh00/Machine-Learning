#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Minzhang Li
# FILE: F:\MyGithubs\Machine-Learning\Supervised-Learning\Nearest-Neighbors\code\KNN_simple.py
# DATE: 2020/08/02 Sun
# TIME: 22:27:09

# DESCRIPTION: This is the simplest way implementing KNN algorithm.

import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from sklearn.datasets import  make_classification
from sklearn import  neighbors, datasets
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime
# create classifier
class KNeighborsClassifier_simple(object):
    def __init__(self, n_neighbors = 15, weights = 'distance'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        # original data
        self.X = None
        self.y = None
        # combination of samples
        self.samples = None
        self.dim = None

    def __distance(self, x, p):
        '''distance algorithm'''
        res = 0
        for i in range(self.dim):
            res += (x[i] - p[i]) ** 2
        # n root
        return np.power(res, 1 / self.dim)

    def __output(self, neighbors):
        '''When K neighbors array already, output labels'''
        # given neigbors array, output prediected label
        n_labels = len(set(self.y))
        labels = np.zeros(n_labels)
        # count num based on distance
        for i in neighbors:
            if i[-1] != 0: labels[int(i[-2])] += 1/ i[-1] 
            else: labels[int(i[-2])] += 1e9
        return np.argmax(labels)   

    def __predict_point(self, p):
        '''predict one point'''
        # brute-forcely calculate distance
        for i in range(len(self.samples)):
            self.samples[i][-1] = self.__distance(self.samples[i], p)
        # sort and find k neighbors 
        ord = list(self.samples.copy())
        ord.sort(key = lambda x: x[-1])
        ord = np.array(ord)
        neighbors = ord[:self.n_neighbors]
        # predict the res
        return self.__output(neighbors)

    def fit(self, X, y):
        assert len(X) == len(y) and len(X) != 0
        # X, y
        self.X = X
        self.y = y
        # get dim
        self.dim = len(X[0])
        self.samples = np.c_[X, y]
        distances = np.zeros(len(self.samples), dtype = float)
        # add one more dim in samples for distance
        self.samples = np.c_[self.samples, distances]
    def predict(self, P):
        res = np.zeros(len(P))
        for i, p in enumerate(P):
            res[i] = self.__predict_point(p)
        return res
    def score(self, X, y):
        '''score without predicted data'''
        y_predicted = self.predict(X)
        res = np.sum(y_predicted == y ) / len(y)
        return res

# same as KNN_sklearn_test.py
def KNN_sklearn(X, y):
    # -- split test set from train set
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    # 2. generate to-be-classified data (predicted data)
    # -- get the range of to-be-classified point 
    x_min, x_max = x_train[:,0].min() - 1, x_train[:,0].max() + 1
    y_min, y_max = x_train[:,1].min() - 1, x_train[:,1].max() + 1
    # -- mesh 
    h = .1 # step size for mesh
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
    plt.figure(figsize = (40, 20))
    for i in range(len(K)):
        plt.subplot(3,3,i+1)
        plt.pcolormesh(xx, yy, res[i].reshape(xx.shape), cmap = color_light)
        plt.scatter(x_train[:,0], x_train[:,1], c=y_train, cmap=color_bold)
        # add title
        plt.title(f'sklearn-{i+1}NN')
    plt.savefig('../img/fig3.png', format='png')

    # 7. plot the scores
    plt.figure()
    plt.title('K-NN varying number of neighbors')
    plt.plot(K, train_accuracy, label = 'Training accuracy of sklearn version')
    plt.plot(K, test_accuracy,  label = 'Testing accuracy of sklearn version')
    plt.legend()
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy')
    plt.savefig('../img/fig4.png', format='png')

def KNN_simple(X, y):
    # -- split test set from train set
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    # 2. generate to-be-classified data (predicted data)
    # -- get the range of to-be-classified point 
    x_min, x_max = x_train[:,0].min() - 1, x_train[:,0].max() + 1
    y_min, y_max = x_train[:,1].min() - 1, x_train[:,1].max() + 1
    # -- mesh 
    h = .1 # step size for mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # -- flatten and pair up
    P = np.c_[xx.ravel(),yy.ravel()] # P is an vector of points 
    # To find the best K we should test the result at test set:
    K = np.arange(1,10) # [1,10)
    test_accuracy = np.zeros(len(K))
    train_accuracy = np.zeros(len(K))
    res = []
    plt.figure(figsize = (40, 20))
    for i, k in enumerate(tqdm(K)):
        # 3. generate classifier
        clf = KNeighborsClassifier_simple(n_neighbors=k, weights='distance')
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
        plt.title(f'simple-{i+1}NN')
    plt.savefig('../img/fig5.png', format='png')

    # 7. plot the scores
    plt.figure()
    plt.title('K-NN varying number of neighbors')
    plt.plot(K, train_accuracy, label = 'Training accuracy of simple version')
    plt.plot(K, test_accuracy,  label = 'Testing accuracy of simple version')
    plt.legend()
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy')
    plt.savefig('../img/fig6.png', format='png')

def main():
    # 1. generate samples. 200 samples for which are of two dims, 3-classes labels. note that dont mess up 'class' with 'cluster'
    X, y = make_classification(n_samples=200, n_features=2 ,n_redundant=0, n_clusters_per_class=1, n_classes=3)
    start = datetime.now()
    KNN_sklearn(X, y)
    end = datetime.now()
    print(f"sklearn-version time: {end - start}")
    start = datetime.now()
    KNN_simple(X, y)
    end = datetime.now()
    print(f"simple-version time: {end - start}")

if __name__ == '__main__':
    main()