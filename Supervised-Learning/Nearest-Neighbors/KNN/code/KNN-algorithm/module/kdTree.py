class Node(object):
    def __init__(self, feature = None, father = None, l_node = None, r_node = None, div = 0, visited = False):
        self.feature = feature
        self.father = father
        self.l_node = l_node
        self.r_node = r_node
        self.div = div
        self.visited = visited

class kdTree(object):
    def __init__(self, X, dim):
        assert X is not None and len(X) != 0
        self.root = Node(div = 0)
        self.dim = dim
        self.samples = X
        self.__build(self.root, list(self.samples))
    def __build(self, node, X):
        # return conditions
        if len(X) == 1: node.feature = X[0]; return
        X.sort(key = lambda x: x[node.div]) # note: X is a list before __build calls.
        # update feature
        node.feature = X[len(X) // 2]
        # create nodes: leaves do not have children
        if len(X[:len(X)//2]):
            node.l_node = Node(father = node, div = (node.div+1) % self.dim)
            self.__build(node.l_node, X[:len(X)//2])
        if len(X[len(X)//2 + 1:]):
            node.r_node = Node(father = node, div = (node.div+1) % self.dim)
            self.__build(node.r_node, X[len(X)//2 + 1:])
        return
    def search(self, p, node):
        if p[node.div] < node.feature[node.div]:
            if node.l_node is not None: return self.search(p, node.l_node)
            return node
        if p[node.div] >= node.feature[node.div]:
            if node.r_node is not None: return self.search(p, node.r_node)
            return node
            
