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
        self.root = Node() # default value
        self.X = None # the last dim is label, please refer to def of 'fit'
    def fit(self, X:np.ndarray, y:np.ndarray):
        # it is required that X and y are np.ndarray
        self.X = np.c_[self.X, self.y]
    def __build(self, node, X):
        '''Recursively expand(train) the tree.'''
        # 1.
        
        
        