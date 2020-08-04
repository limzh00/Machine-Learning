#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Minzhang Li
# FILE: F:\MyGithubs\Machine-Learning\Supervised-Learning\Nearest-Neighbors\code\KNN_kdTree.py
# DATE: 2020/08/03 Mon
# TIME: 16:42:57

# DESCRIPTION: kd-Tree KNN

import numpy as np 
import pandas as pd 
import scipy as sp 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from datetime import datetime
from tqdm import tqdm
from sklearn.datasets import make_classification
from sklearn import datasets, neighbors
from module.kdTree import kdTree
from sklearn.model_selection import train_test_split

class KNeighborsClassifier(object):
    def __init__(self, n_neighbors, weights = 'distance'):
        self.samples = None
        self.dim = None
        self.tree = None
        self.K = n_neighbors
        self.weights = weights
        # original data
        self.X = None
        self.y = None
        # k neighbors array
        self.neighbors = []
    def __decision_rules(self, neighbors):
        '''For a to-be-classified point, this function decides its class based on k-neighbors of it. 
        return the label predicted.'''
        labels = np.zeros(len(set(self.y)))
        for i, neighbor in enumerate(neighbors):
            if neighbor[-1] == 0: return neighbor[-2]
            labels[int(neighbor[-2])] += 1 / neighbor[-1]
        return np.argmax(labels)       
    def __distance(self, x, p):
        # it is required the x and p are type of ndarray
        res = np.sum((x - p)**2)
        return np.power(res, 1 / self.dim)
    def __find_k_neighbors(self, node, neighbors, p):
        # note that, neighbors should be in order now.

        # Given a node, we technically do three things:
        # 1. check whether we should involve this node in
        # 2. figure out in which node we forward our recursion, node.father or node.child?
        # 3. check ending condition

        # firstly, set node visited.
        node.visited = True
        node.feature[-1] = self.__distance(node.feature[:-2], p)
        # 1. neighbors updates
        # if k < self.K, we just add it into neighbors
        if len(self.neighbors) < self.K: self.neighbors.append(node.feature)
        # else, check whether involve it in neighors
        elif node.feature[-1] < self.neighbors[-1][-1]: 
            self.neighbors[-1] = node.feature
            self.neighbors.sort(key = lambda x: x[-1])
        # 2. node to be forwarded towards
        # if neighbors is full
        # enter node father
        if abs(node.feature[node.div] - p[node.div]) >= self.neighbors[-1][-1] and len(self.neighbors) >= self.K:
            if node.father is not None and node.father.visited is False: 
                self.__find_k_neighbors(node.father, self.neighbors, p)
            node.visited = False
            return
        else:
            if node.l_node is not None and node.l_node.visited is False: 
                self.__find_k_neighbors(node.l_node, neighbors, p)
            if node.r_node is not None and node.r_node.visited is False: 
                self.__find_k_neighbors(node.r_node, neighbors, p)
            if node.father is not None and node.father.visited is False: 
                self.__find_k_neighbors(node.father, neighbors, p)
            # note: before return, set visited false
            node.visited = False
            return
    def __predict_point(self, p):
        '''P is just a point, only one data to be classified'''
        # note: neighbors should always be a list.
        self.neighbors = []
        node = self.tree.search(p, self.tree.root)
        self.__find_k_neighbors(node, self.neighbors, p)
        # print(self.neighbors)
        return self.__decision_rules(self.neighbors)
    def fit(self, X, y):
        assert len(X) == len(y) and len(X)
        # combinate X and y, add one more dim for distance and done.
        # note: this operation can avoid the error that X is one dimensional
        self.samples = np.c_[X, y]
        self.samples = np.c_[self.samples, np.zeros(len(self.samples))]
        # dim
        self.dim = len(X[0])
        # kd-Tree
        self.tree = kdTree(self.samples, self.dim)
        # original data
        self.X = X
        self.y = y
        return         
    def predict(self, P):
        '''Given P, output prediected result Z.'''
        # note that P is required to be one-dimensional vector
        # the result initialization
        res = np.zeros(len(P))    
        for i, p in enumerate(P):
            res[i] = self.__predict_point(p)
        return res
    def score(self, X, y):
        y_predicted = self.predict(X)
        return np.sum(y_predicted == y) / len(y)

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
    plt.title("sklearn-version")
    for i in range(len(K)):
        plt.subplot(3,3,i+1)
        plt.pcolormesh(xx, yy, res[i].reshape(xx.shape), cmap = color_light)
        plt.scatter(x_train[:,0], x_train[:,1], c=y_train, cmap=color_bold)
        # add title
        plt.title(f'sklearn-{i+1}NN')
    plt.savefig('./fig_sklearn_7.png', format='png')

    # 7. plot the scores
    plt.figure()
    plt.title('K-NN varying number of neighbors: sklearn')
    plt.plot(K, train_accuracy, label = 'Training accuracy of sklearn version')
    plt.plot(K, test_accuracy,  label = 'Testing accuracy of sklearn version')
    plt.legend()
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy')
    plt.savefig('./fig_sklearn_8.png', format='png')


def KNN_kdTree(X, y):
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
        clf = KNeighborsClassifier(n_neighbors=k, weights='distance')
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
    plt.title('kdTree-version')
    for i in range(len(K)):
        plt.subplot(3,3,i+1)
        plt.pcolormesh(xx, yy, res[i].reshape(xx.shape), cmap = color_light)
        plt.scatter(x_train[:,0], x_train[:,1], c=y_train, cmap=color_bold)
        # add title
        plt.title(f'kdTree-{i+1}NN')
    plt.savefig('./fig_kdTree_7.png', format='png')

    # 7. plot the scores
    plt.figure()
    plt.title('K-NN varying number of neighbors: kdTree')
    plt.plot(K, train_accuracy, label = 'Training accuracy of kdTree version')
    plt.plot(K, test_accuracy,  label = 'Testing accuracy of kdTree version')
    plt.legend()
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy')
    plt.savefig('./fig_kdTree_8.png', format='png')

def main():
    # 1. generate samples. 200 samples for which are of two dims, 3-classes labels. note that dont mess up 'class' with 'cluster'
    X, y = make_classification(n_samples=200, n_features=2 ,n_redundant=0, n_clusters_per_class=1, n_classes=3)
    start = datetime.now()
    KNN_kdTree(X, y)
    end = datetime.now()
    print(f"kdTree-version time: {end - start}")
    start = datetime.now()
    KNN_sklearn(X, y)
    end = datetime.now()
    print(f"sklearn-version time: {end - start}")


if __name__ == "__main__":
    main()