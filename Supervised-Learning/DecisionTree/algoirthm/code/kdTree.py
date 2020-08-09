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
    def __init__(self, n_neighbors = 5):
        self.root = Node(r = 0)
        self.X = None
        self.y = None
        self.samples = None
        self.dim = None
        self.neighbors = []
        self.K = n_neighbors
        
    def __distance(self, x, p):
        return np.power(np.sum((x[:-2] - p)**self.dim), 1 / self.dim)
    def __search(self, node, p):
        if p[node.r] > node.feature[node.r]:
            if node.r_node is None: return node
            return self.__search(node.r_node, p)
        else:
            if node.l_node is None: return node
            return self.__search(node.l_node, p)
        
    def __predict_point(self, node, p):
        node.visited = True
        node.feature[-1] = self.__distance(node.feature, p)
        if len(self.neighbors) < self.K:
            self.neighbors.append(node.feature)
            if node.l_node is not None and node.l_node.visited is not True:
                self.__predict_point(node.l_node, p)
            if node.r_node is not None and node.r_node.visited is not True:
                self.__predict_point(node.r_node, p)
            if node.father is not None and node.father.visited is not True:
                self.__predict_point(node.father, p)
            node.visited = False
            return 
        else:
            self.neighbors.sort(key = lambda x: x[-1])
            if self.neighbors[-1][-1] <= abs(node.feature[node.r] - p[node.r]):
                node.visited = False
                return 
            else:
                if self.neighbors[-1][-1] > node.feature[-1]:
                    self.neighbors[-1] = node.feature
                    if node.l_node is not None and node.l_node.visited is not True:
                        self.__predict_point(node.l_node, p)
                    if node.r_node is not None and node.r_node.visited is not True:
                        self.__predict_point(node.r_node, p)
                    if node.father is not None and node.father.visited is not True:
                        self.__predict_point(node.father, p)
                node.visited = False
                return 
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
            self.__build(node.r_node, X[len(X)//2 + 1:])

    def fit(self, X, y):
        self.samples = np.c_[X, y]
        self.samples = np.c_[self.samples, np.zeros(len(self.samples))]
        self.X = X
        self.y = y
        self.dim = len(X[0])
        self.__build(self.root, self.samples)
    def predict(self, P):
        res = np.zeros(len(P))
        for i, p in enumerate(tqdm(P)):
            self.__predict_point(self.__search(self.root, p), p)
            res[i] = Counter(np.array(self.neighbors)[:,-2]).most_common(1)[0][0]
            print(res[i])
        return res
    def score(self, X, y):
        y_predicted = self.predict(X)
        print(y_predicted, y)
        return np.sum(y_predicted == y) / len(y)
def preprocess(total_info:dict) -> tuple:
    return (total_info['data'], total_info['target'], total_info['target_names'])

def main():
    # 1. load data
    (X, y, labels_name) = preprocess(datasets.load_iris())
    # 2. split X and y -- construct test cases and train cases
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    # 3. construct decision tree and fit the data
    clf = kdTree(10)
    clf.fit(x_train, y_train)
    # 4. predict and score
    accuracy = clf.score(x_test, y_test)
    # 5. draw this tree
    # plt.figure(figsize=(10,10))
    # plt.title('sklearn-DecisionTree')
    # tree.plot_tree(clf)
    # plt.savefig('../img/fig1.png')
    print(accuracy)
    
if __name__ == '__main__':
    main()