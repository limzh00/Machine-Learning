import numpy as np 
import pandas as pd 
import scipy as sp
import matplotlib.pyplot as plt
# store the model
import pickle 
from tqdm import tqdm
from collections.abc import Iterable
from collections import Counter
from sklearn import tree, datasets
from sklearn.model_selection import train_test_split

class Node(object):
    def __init__(self, father:Node = None, l_node:Node = None, r_node:Node = None,
                 r:int = None, entropy:float = None, n_samples:int = None, values:list = None):
        self.father = father
        self.l_node = l_node
        self.r_node = r_node
        self.r = r
        self.entropy = entropy
        self.n_samples = n_samples
        self.values = values

class DecisionTreeClassifier(object):
    def __init__(self, criterion = 'entropy'):
        self.criterion = criterion
        self.root = None # default value
        self.X = None # the last dim is label, please refer to def of 'fit'
        self.n_labels = None
    def fit(self, X:np.ndarray, y:np.ndarray):
        # it is required that X and y are np.ndarray
        self.X = np.c_[X, y]
        self.n_labels = len(set(y))
        print("The numbers of labels: ", self.n_labels)
        # build the tree
        self.root = Node(n_samples = len(self.X))
        self.__build(self.root, self.X)
    def __cal_entropy(self, X):
        X = np.array(X)
        Px = np.zeros(self.n_labels, dtype = int)
        vals = Counter(X[:,-1]).most_common()
        vals = np.array(vals, dtype=int)
        for i, j in vals:
            Px[i] = j
        Px = Px / np.sum(Px)
        return - np.sum(Px * np.log2(Px))
    
    def __ID3(self, subset1, subset2, original_set):
        D = self.__cal_entropy(original_set)
        D1 = self.__cal_entropy(subset1)
        D2 = self.__cal_entropy(subset2)
        return D - (len(subset1) / len(original_set) * D1+ len(subset2) / len(original_set) * D2)

        
    def __build(self, node, X):
        '''Recursively expand(train) the tree.'''
        # 1. update the node attributes
        # -- node.values
        node.values = np.zeros(self.n_labels, dtype = int)
        vals = Counter(X[:,-1]).most_common()
        vals = np.array(vals, dtype = int)
        print(vals)
        for i, j in vals:
            node.values[i] = j
        # -- node.n_samples
        node.n_samples = len(X)
        # -- node.entropy
        node.entropy = self.__cal_entropy(X)
        # 2. divide
        # -- total-dimensions
        n_dim = len(X[0]) - 1
        # -- list
        X = X.tolist(); div = None; val = None; max_entropy = None;
        for i in range(n_dim):
            X.sort(key = lambda x:x[i])
            # get candidates
            for j in range(1,len(X))：
                if( j/len(X) * self.__cal_entropy(X[:j]) +  (len(X)-j)/len(X) * self.__cal_entropy(X[j:])):
                    pass

        
        