from kdTree import kdTree
import numpy as np
class KNeighborsClassifier_kdTree(object):
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