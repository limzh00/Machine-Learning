import numpy as np

class kdNode:
    # constructor
    def __init__(self, data = None, depth = 0, div = 0, father = None, left = None, right = None):
        self.data = data # n-dimention vector
        self.depth = depth # depth in tree
        self.div = div # divided dimention
        self.father = father # father
        self.left = left # left child
        self.right = right # right child

    # build a kdTree   
    def build_tree(self, dataset, depth):
        # return if no data
        if not len(dataset):
            return
        # get the divided dimention by depth
        dim = len(dataset[0])
        div = depth % dim
        # sort the data according to the number in divided dimention
        dataset.sort(key = lambda x:x[div])
        # find the median point
        median_index = len(dataset) // 2
        self.data = dataset[median_index]
        # build subtrees
        self.left = kdNode(None, depth + 1, div, self)
        self.right = kdNode(None, depth + 1, div, self)
        self.left.build_tree(dataset[:median_index], depth + 1)
        self.right.build_tree(dataset[median_index + 1:], depth + 1)
    
    # an preorder traversal function to check the tree
    def pre_traversal(self):
        if not self.data:
            return
        print(self.data, self.depth)
        self.left.pre_traversal()
        self.right.pre_traversal()

class kdTree:
    # constructor
    def __init__(self, data = None, depth = 0, father = None, left = None, right = None):
        self.root = kdNode(data, depth, father, left, right)
    
    # build a kdTree
    def build_tree(self, dataset, depth = 0):
        self.root.build_tree(dataset, depth)
    
    # preorder traversel
    def tree_pre_traversal(self):
        self.root.pre_traversal()

# find the max distance in current neighbors
def curr_max_dist(point, l):
    # initialize the parameters
    total = len(l)
    curr_max = 0
    max_dist_index = -1
    for i in range(total):
        # compute the distance of each neighbor
        dist = (( (np.array(point) - np.array(l[i]) )**2).sum())**0.5
        # find the maximum
        if dist > curr_max:
            # update the maximum and the index
            curr_max = dist
            max_dist_index = i

    return curr_max, max_dist_index

# find k closest neighbors
def find_k_neighbors(point, root, k, neighbors):
    # start from the root node
    curr_root = root
    # continue until reaching the bottom node
    while curr_root.left or curr_root.right:
        if point[curr_root.div] < curr_root.data[curr_root.div]:
            curr_root = curr_root.left
        else:
            curr_root = curr_root.right
    # add current node to neighbors if it isn't full
    if len(neighbors) < k:
        neighbors.append(curr_root.data)
    # replace the node with the maximum distance if the current one is closer
    else:
        max_dist, max_dist_index = curr_max_dist(point, neighbors)
        curr_dist = (( (np.array(point) - np.array(curr_root.data) )**2).sum())**0.5
        if(max_dist > curr_dist):
            neighbors[max_dist_index] = curr_root.data
    #
    #
    #
    #
    # how to mark the visited points?
    #
    #
    #
    #


if __name__ == "__main__":
    # check the tree-building
    kd_tree = kdTree()
    points = [[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]
    kd_tree.build_tree(points)
    kd_tree.tree_pre_traversal()

    # neighbors = []
    # find_k_neighbors(point, kd_tree.root, neighbors)

    