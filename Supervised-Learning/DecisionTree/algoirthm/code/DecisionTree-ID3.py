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
    def __init__(self, father = None, l_node = None, r_node = None,
                 r:tuple = None, entropy:float = None, n_samples:int = None, values:list = None):
        self.father = father
        self.l_node = l_node
        self.r_node = r_node
        self.r = r
        self.entropy = entropy
        self.n_samples = n_samples
        self.values = values

class Node(object):
    def __init__(self, father = None, l_node = None, r_node = None, split = None, n_samples = None, values = None, entropy = None, label = None):
        self.split = split
        self.values = values
        self.l_node = l_node
        self.r_node = r_node
        self.father = father
        self.n_samples = n_samples
        self.entropy = entropy
        self.label = label

class DecisionTreeClassifier(object):
    def __init__(self, criterion = 'entropy'):
        self.root = Node()
        self.X = None
        self.n_dim = None
        self.n_labels = None
    def fit(self, X, y):
        self.X = np.c_[X, y]
        self.n_dim = len(X[0])
        self.n_labels = len(set(y))
        self.__build(self.root, self.X)
    def __extract(self, X):
        res = np.zeros(self.n_labels)
        X = Counter(X.tolist()).most_common()
        for i, j in X:
            res[int(i)] = j
        return res
    def __cal_entropy(self, X):
        X = Counter(X.tolist()).most_common()
        X = np.array(X)[:,-1]
        res = X / np.sum(X)
        return - np.sum(res * np.log2(res))
        
    def __build(self, node, X):
        node.values = self.__extract(X)
        node.n_samples = len(X)
        node.entropy = self.__cal_entropy(X)
        node.label = np.argmax(node.values)
        if node.entropy <= 1e-5: return
        tmpX = X.tolist()
        memo = []
        for i in range(self.n_dim):
            tmpX.sort(key = lambda x:x[i])
            for j in range(0, len(tmpX)):
                sub1_entropy = self.__cal_entropy(X[:j])
                sub2_entropy = self.__cal_entropy(X[j:])
                gain = node.entropy - (j / node.n_samples * sub1_entropy + (node.n_samples - j) / node.n_samples * sub2_entropy)
                val = (tmpX[j] + tmpX[j-1])/2
                memo.append((i,j,val,gain))
        memo.sort(key = lambda x:x[-1])
        dim, j, val , gain = memo[-1]
        node.split = (dim, val)
        X = X.tolist()
        X.sort(key = lambda x:x[dim])
        sub1 = np.array(X[:j])
        sub2 = np.array(X[j:])
        if len(sub1):
            node.l_node = Node(father = node)
            self.__build(node.l_node, sub1)
        if len(sub2):
            node.r_node = Node(father = node)
            self.__build(node.r_node, sub2)
    def __check(node, p):
        dim, val = node.split
        if node.l_node is not None and p[dim] <= val:
            return self.__check(node.l_node)
        elif node.r_node is not None and p[dim] > val:
            return self.__check(node.r_node)
        return node.label
    def __predict_point(self, p):
        return self.__check(self.root, p)
    def predict(self, P):
        res = np.zeros(len(P))
        for i, p in enumerate(tqdm(P)):
            res[i] = self.__predict_point(p)
        return res
    def score(self, X, y):
        y_predicted = self.predict(X)
        return np.sum(y_predicted == y) / len(y)
                
        
        

def preprocess(total_info:dict) -> tuple:
    return (total_info['data'], total_info['target'], total_info['target_names'])

def main():
    # 1. load data
    (X, y, labels_name) = preprocess(datasets.load_iris())
    # 2. split X and y -- construct test cases and train cases
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    # 3. construct decision tree and fit the data
    clf = DecisionTreeClassifier()
    clf1 = tree.DecisionTreeClassifier(criterion='entropy')
    clf.fit(x_train, y_train)
    clf1.fit(x_train, y_train)
    # 4. predict and score
    accuracy = clf.score(x_train, y_train)
    accuracy1 = clf1.score(x_train, y_train)
    # 5. draw this tree
    # plt.figure(figsize=(10,10))
    # plt.title('sklearn-DecisionTree')
    # tree.plot_tree(clf)
    # plt.savefig('../img/fig1.png')
    print(accuracy, accuracy1)
    
if __name__ == '__main__':
    main()



        
        