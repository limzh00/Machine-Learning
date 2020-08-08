import numpy as np 
import pandas as pd 
import scipy as sp 
import  matplotlib.pyplot as plt 
from sklearn import datasets, neighbors
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections.abc import Iterable
from collections import Counter

class Node(object):
    def __init__(self, father = None, l_node = None, r_node = None, feature = None, r = 0, visited = False):
        self.father = father
        self.l_node = l_node
        self.r_node = r_node
        self.feature = feature
        self.r = r
        self.visited = visited

class kdTree(object):
    def __init__(self):
        self.root = Node(r = 0)
        self.X = None
        self.y = None
        self.samples = None
        self.dim = None
    def __distance(self, x, p):
        return np.power(np.sum((x - p)**self.dim), 1 / self.dim)
    def __search(self, node, p):
        
    def __predict_point(self, p):
        pass

    def __build(self, node, X):
        X = X.tolist()
        X.sort(key = lambda x: x[node.r])
        X = np.array(X)
        node.feature = X[len(X) // 2]
        if len(X[:len(X)//2]) != 0:
            node.l_node = Node(father = node, r = (node.r + 1) % self.dim)
            self.__build(node.l_node, X[:len(X)//2])
        if len(X[len(X)//2 + 1 : ]) != 0:
            node.r_node = Node(father = node, r = (node.r + 1) % self.dim)
            self.__build(node.e_node, X[len(X)//2 + 1:])
    def fit(self, X, y):
        self.samples = np.c_[X, y]
        self.X = X
        self.y = y
        self.dim = len(y)
        self.__build(self.root, self.samples)
    def predict(self, P):
        res = np.zeros(len(P))
        for i, p in enumerate(tqdm(P)):
            res[i] = self.__predict_point(p)
        return res