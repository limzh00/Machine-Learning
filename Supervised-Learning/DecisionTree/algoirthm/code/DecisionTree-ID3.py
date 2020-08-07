import numpy as np 
import pandas as pd 
import scipy as sp
import matplotlib.pyplot as plt
# store the model
import pickle 
from collections import Iterable, Counter


class Node(object):
    def __init__(self, div = None, value = None, l_node = None, r_node = None):
        self.div = div
        self.value = value
        self.l_node = l_node
        self.r_node = r_node


class DecisionTree(object):
    def __init__(self):
        self.root = Node()
        # original data
        self.X = None
        self.y = None
        self.dim:int = None
        # combination of original data 
        self.samples = None
    def __cal_entropy(self, samples) -> float:
        '''Given dataset X, calculate its information entropy.'''
        # count numbers
        nums = Counter(samples[:,-1]).most_common()
        # 1. get emperical probability distribution
        P = np.array(nums)[:,-1] / len(nums) 
        # 2. entropy
        return -np.sum(P * np.log2(P))
        
    def __build(self, node, X):
        pass
    def fit(self, X, y):
        '''fit the trained data'''
        # it is required that X and y are vector or just an instance of data
        X = np.array(X); y = np.array(y)
        # 1. set priginal data as attributes
        self.X = X
        self.y = y
        self.dim = len(set(self.y)) # no duplicated data
        # 2. combine them
        self.samples = np.c_[self.X, self.y]
        # 3. using data training a tree and store as a trained model
        # -- 3.1 train the tree
        self.__build(self.node, self.samples)
        # -- 3.2 save the model file
        with open('./model/Iris.csv', 'wb') as f:
            pickle.dump(self, f, protocol=0)
            f.close()
    def predict(self):
        pass
    def score(self):
        pass
    def plot_tree(self):
        pass

        