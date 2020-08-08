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
    def __init__(self, div = None, value = None, l_node = None, r_node = None, father = None):
        self.div = div
        self.value = value
        self.is_continuous = None
        self.l_node = l_node
        self.r_node = r_node
        self.father = father
        self.label = None

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
        P = np.array(nums)[:,-1] / len(samples)
        # 2. entropy
        return -np.sum(P * np.log2(P))
    
    def __ID3_rule(self, D, D_new, max_gain):
        if D - D_new > max_gain:
            return (True, D-D_new)
        return (False, max_gain)

    def __build(self, node, samples):
        # update node.label
        node.label = Counter(samples[:,-1]).most_common(1)[0][0]
        # base entropy
        D = self.__cal_entropy(samples)
        # iterate over all dimensions, find the best one to split on
        (sub1, sub2) = (None, None)
        node.value = None
        max_gain = 0
        for i in range(self.dim):
            # -- put samples in order on dim basis
            samples = samples.tolist()
            samples.sort(key = lambda x: x[i])
            samples = np.array(samples) # maintain np.array
            # -- condidates value
            val = np.array(list(set(samples[:,i])))
            # check continuity
            # if len(candidates) / len(self.samples[:,i]) >= 1 / 5, continuous
            if len(val) / len(samples[:,i]) >= 0: is_continuous = True
            else: is_continuous = False
            # process candidates if continuous
            if is_continuous:
                val = np.array([(val[i] + val[i-1])/2 for i in range(1, len(val))])
            # iterate over all candidated values, find the best one to split on
            for j, v in enumerate(val):
                # split on value v
                if is_continuous:
                    subset1 = samples[samples[:,i] >= v]
                    subset2 = samples[samples[:,i] <  v]
                    # print(subset1)
                else:
                    subset1 = samples[samples[:,i] == v]
                    subset2 = samples[samples[:,i] != v]
                # new entropy
                D_new = len(subset1) / len(samples) * self.__cal_entropy(subset1) + len(subset2) / len(samples) * self.__cal_entropy(subset2)
                (flag, max_gain) = self.__ID3_rule(D, D_new, max_gain)
                if flag: 
                    sub1 = subset1
                    sub2 = subset2
                    # store node info
                    node.value = v
                    node.is_continuous = is_continuous
                    node.div = i   

        if len(sub1) != 0 and self.__cal_entropy(sub1) > 1e-1:
            node.l_node = Node(father = node)
            self.__build(node.l_node, sub1)
        if len(sub2) != 0 and self.__cal_entropy(sub2) > 1e-1:
            node.r_node = Node(father = node)
            self.__build(node.r_node, sub2)

    def __predict_point(self, x):
        node = self.root
        res = None
        while True:
            res = node.label
            if node.is_continuous:
                if x[node.div] >= node.value: node = node.l_node
                else: node = node.r_node
            else:
                if x[node.div] == node.value: node = node.l_node
                else: node = node.r_node
            # if node == None, it means we have reached the end
            if node is None:
                return res

    def fit(self, X, y):
        '''fit the trained data'''
        # it is required that X and y are vector or just an instance of data
        X = np.array(X); y = np.array(y)
        # 1. set priginal data as attributes
        self.X = X
        self.y = y
        # 2. combine them and remove duplicated ones
        self.samples = np.c_[self.X, self.y]
        self.dim = len(set(self.y)) # no duplicated data 
        # 3. using data training a tree and store as a trained model
        # -- 3.1 train the tree
        self.__build(self.root, self.samples)
        # -- 3.2 save the model file
        with open('./model/Iris.csv', 'wb') as f:
            pickle.dump(self, f, protocol=0)
            f.close()

    def predict(self, P):
        res = np.zeros(len(P))
        for i, p in enumerate(tqdm(P)):
            res[i] = self.__predict_point(p)
        return res

    def score(self, X, y):
        y_predicted = self.predict(X)
        return np.sum(y_predicted == y) / len(y)

    def plot_tree(self):
        pass

def preprocess(total_info:dict) -> tuple:
    return (total_info['data'], total_info['target'], total_info['target_names'])

def main():
    # 1. load data
    (X, y, labels_name) = preprocess(datasets.load_iris())
    # 2. split X and y -- construct test cases and train cases
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    # 3. construct decision tree and fit the data
    clf = DecisionTree()
    clf.fit(x_train, y_train)
    # 4. predict and score
    accuracy = clf.score(x_test, y_test)
    # 5. draw this tree
    # plt.figure(figsize=(10,10))
    # plt.title('sklearn-DecisionTree')
    # tree.plot_tree(clf)
    # plt.savefig('../img/fig1.png')
    print(accuracy)
if __name__ == "__main__":
    main()
        