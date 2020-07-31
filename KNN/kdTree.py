import numpy as np
from collections import Iterable

class kdNode(object):
    # constructor
    def __init__(self, feature = None, div = 0, father = None, left = None, right = None):
        self.feature = feature # n-dimension vector
        self.div = div # divided dimension
        self.father = father # father
        self.left = left # left child
        self.right = right # right child
        self.isvisited = False



class kdTree(object):
    # constructor
    def __init__(self, data = ()):
        assert isinstance(data, Iterable)
        self.root = kdNode()
        self.data = data
    # build a kdTree   
    def build(self, node, dataset, div): 
        # return if no data
        if not len(dataset): return
        # sort the data according to the number in divided dimention
        dataset.sort(key = lambda x:x[div])
        # find the median point
        median_index = (len(dataset) - 1) // 2
        node.feature = dataset[median_index]
        # build subtrees
        node.left = kdNode(div = (div + 1) % len(self.data[0]), father = node)
        node.right = kdNode(div = (div + 1) % len(self.data[0]), father = node)
        self.build(dataset = dataset[:median_index], node = node.left, div = node.left.div)
        self.build(dataset = dataset[median_index + 1:], node = node.right, div = node.right.div)
    
    # an preorder traversal function to check the tree
    def pre_traversal(self, node):
        if node.feature == None: 
            return
        print(node.feature, node.div)
        self.pre_traversal(node.left)
        self.pre_traversal(node.right)
    
    def find_k_neighbors(self, neighbors, k, point):
        node = self.find_leaf(point)
        self.enter_neighbors(neighbors, k, point, node)


        # end
        while (up == 1)
            if node == self.root: 
                return
            else:
                up = self.up_or_down(neighbors, k, point, node)



    def up_or_down(self, neighbors, k, point, node):
        while node.isvisited == True:
            node = node.father
        self.enter_neighbors(neighbors, k, point, node)
        dist_to_line = abs(point[node.div] - node.feature[node.div]) 
        distances = [self.distance(point, neighbors[i]) for i in range(len(neighbors))]
        max_dist = max(distances)

        
        if dist_to_line >= max_dist and len(neighbors) == k:
            return 1 #up
        else:
            if node.left.isvisited == False:
                find_k_neighbors(neighbors, k, point)


            

    def enter_neighbors(self, neighbors, k, point, node):
        node.isvisited = True
        distances = [self.distance(point, neighbors[i]) for i in range(len(neighbors))]
        max_dist = max(distances)
        max_index = np.argmax(np.array(distances))
        if len(neighbors) < k:
            neighbors.append(node.feature)
        else:
            if self.distance(point, node.feature) < max_dist:
                neighbors[max_index] = node.feature        

    def find_leaf(self, point):
        node = self.root
        while(True):
            if(point[node.div] < node.feature[node.div]):
                if node.left.feature is None: return node
                node = node.left
            else:
                if node.right.feature is None: return node
                node = node.right
    def distance(self, point, feature):
        res = 0
        for i in range(len(point)):
            res += (point[i] - feature[i])**2
        return np.sqrt(res)

if __name__ == "__main__":
    dataset = [(6.27,5.5),(1.24,-2.86),(17.05, -12.79),(-6.88, -5.40),(-2.96, -0.5), (7.75, -22.68),(10.8, -5.03),(-4.6,-10.55), (-4.96, 12.61), (1.75,12.26), (15.31, -13.16), (7.83, 15.70), (14.63, -0.35)]
    # check the tree-building
    kd_tree = kdTree(data = dataset)
    kd_tree.build(kd_tree.root, kd_tree.data, kd_tree.root.div)
    kd_tree.pre_traversal(kd_tree.root)
    node = kd_tree.find_leaf((-1,-5))
    print(node.feature)
