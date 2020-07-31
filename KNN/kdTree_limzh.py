import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
import math
from sklearn.datasets.samples_generator import make_classification
from scipy import stats
from collections import Iterable

class Node(object):
    def __init__(self, l_node = None, r_node = None, feature = None, father = None, r = None, visited = False):
        self.left_node = l_node
        self.right_node = r_node
        self.feature = feature
        self.father = father
        # self.r is the dimension on which the vector is splitted. 
        self.r = r
        self.isvisited = visited

class kdTree(object):

    def __init__(self, X = ()):
        # make sure that X is iterable.
        assert isinstance(X, Iterable)
        self.X = X
        self.root = Node(r = 0)
        # construct the tree.
        self.__build(self.root, X)
    def __build(self, node, X):
        # retrieve the data
        r = node.r
        # return if X[low,high) is empty.
        if not len(X): return
        # get median at r dimension
        list(X).sort(key = lambda x:x[r])
        node.feature = X[(len(X)-1)//2]
        l_node = Node(father = node, r = (r+1)%(len(self.X[0]) - 1))
        r_node = Node(father = node, r = (r+1)%(len(self.X[0]) - 1))
        node.left_node, node.right_node = l_node, r_node
        self.__build(l_node, X[:(len(X)-1)//2])
        self.__build(r_node, X[(len(X)-1)//2+1:])

class KNN_Classifier(kdTree):
    
    def __init__(self, X, y):
        self.X = np.c_[X,y]
        super(KNN_Classifier, self).__init__(self.X)    
    def __search(self, point):
        '''Given a point, return the node it pairs.'''
        node = self.root
        while(True):
            if point[node.r] < node.feature[node.r]:
                if isinstance(node.left_node.feature, type(None)): return node
                node = node.left_node
            else:
                if isinstance(node.right_node.feature, type(None)): return node
                node = node.right_node

    def __dist(self, point, node):
        res = 0
        for i in range(len(point)):
            res += np.sqrt((point[i] - node.feature[i])**2)
        return res

    def __forward(self, point, node):
        '''Given a point and a node, we check whether we take in subtree(if so, right or left?) or father node'''
        # return 1: access father node; return 0: access subtree node.
        # When the node is leaf node, we must access its father.
        if isinstance(node.left_node.feature,type(None)) and isinstance(node.right_node.feature,type(None)):
            return 0
        # Considering in what case the following code works ----> Only when aftering checking points at edges.
        if abs(node.feature[node.r] - point[node.r]) >= self.max[1]: return 0
        else: return 1
    
    def __access(self, point, node, K):
        '''recursion implementation'''
        if node == None: return 
        if isinstance(node.feature,type(None)): return
        if node.isvisited: self.__access(point, node.father, K); return
        # if node is not visited, we check whether add it to the neighbors
        node.isvisited = True
        # if self.num < K, we add it.
        if self.num < K:
            self.neighbors[self.num], self.distance[self.num] = list(node.feature), self.__dist(point, node)
            self.num += 1
        # else, we compare.
        else:
            if self.__dist(point, node) < self.max[1]:
                self.neighbors[self.max[0]] = list(node.feature)
                self.distance[self.max[0]] = self.__dist(point, node)
        # update    
        self.max[0], self.max[1] = np.argmax(self.distance), np.max(self.distance)
        # after updating, we wants to know where to forward our recurrsion, subtree node or father node?
        # it depends on forward()
        if not self.__forward(point, node):
            self.__access(point, node.father, K); return
        else:
            self.__access(point, node.left_node, K)
            self.__access(point, node.right_node,K)
            self.__access(point, node.father, K)
            return 
    def __find_k_neighbors(self, K, point):
        self.neighbors = [Node() for i in range(K)]
        self.distance = np.full(K,1e8)
        self.max = [np.argmax(self.distance), max(self.distance)]
        # the number of neighbors 
        self.num = 0
        self.__access(point, self.__search(point),K)
        self.flush(self.root)
        return self.neighbors, self.distance
    
    def flush(self, node):
        if isinstance(node.feature, type(None)): return
        node.isvisited = False
        self.flush(node.left_node)
        self.flush(node.right_node)

    def train(self, K, X, y, h = .5):
        x_min, x_max = np.array(X)[:,0].min() - 1, np.array(X)[:,0].max() + 1
        y_min, y_max = np.array(X)[:,1].min() - 1, np.array(X)[:,1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        points = np.c_[xx.ravel(), yy.ravel()]
        res = np.zeros(shape = (len(np.arange(x_min, x_max, h)),len(np.arange(y_min, y_max, h))))
        m = 0
        for i in range(len(res)):
            for j in range(len(res[0])):
                neighbors, distance = np.array(self.__find_k_neighbors(K,points[m]))
                label = np.zeros(K)
                for k in range(len(distance)):
                    label[int(neighbors[k][-1])] += 1 / distance[k] 
                # res[i][j] = stats.mode(neighbors[:,-1])[0][0]
                res[i][j] = np.argmax(label)
                m += 1
        return res
        
        
        
        
if __name__ == "__main__":
    # dataset = [[6.27,5.5],[1.24,-2.86],[17.05, -12.79],[-6.88, -5.40],[-2.96, -2.5], [7.75, -22.68],[10.8, -5.03],[-4.6,-10.55], [-4.96, 12.61], [1.75,12.26], [15.31, -13.16], [7.83, 15.70], [14.63, -0.35]]
    # y = [0,1,2,1,1,2,2,1,0,0,2,0,2]
    dataset, y = make_classification(n_samples = 200, n_features=2, n_redundant = 0, 
                    n_clusters_per_class=1, n_classes = 3)
    tree = KNN_Classifier(dataset, y)
    h = .05
    res = tree.train(K = 20, X = dataset, y = y, h = h)
    # create color maps
    x_min, x_max = np.array(dataset)[:,0].min() - 1, np.array(dataset)[:,0].max() + 1
    y_min, y_max = np.array(dataset)[:,1].min() - 1, np.array(dataset)[:,1].max() + 1
    # data
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    res = res.reshape(xx.shape)


    cmap_light = ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000','#003300','#0000FF'])
    plt.pcolormesh(xx,yy,res,cmap = cmap_light)
    plt.scatter(np.array(dataset)[:,0], np.array(dataset)[:,1], c = y, cmap=cmap_bold)
    plt.show()

    
    
        
    
