class KNeighborsClassifier_simple(object):
    def __init__(self, n_neighbors = 15, weights = 'distance'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        # original data
        self.X = None
        self.y = None
        # combination of samples
        self.samples = None
        self.dim = None

    def __distance(self, x, p):
        '''distance algorithm'''
        res = 0
        for i in range(self.dim):
            res += (x[i] - p[i]) ** 2
        # n root
        return np.power(res, 1 / self.dim)

    def __output(self, neighbors):
        '''When K neighbors array already, output labels'''
        # given neigbors array, output prediected label
        n_labels = len(set(self.y))
        labels = np.zeros(n_labels)
        # count num based on distance
        for i in neighbors:
            if i[-1] != 0: labels[int(i[-2])] += 1/ i[-1] 
            else: labels[int(i[-2])] += 1e9
        return np.argmax(labels)   

    def __predict_point(self, p):
        '''predict one point'''
        # brute-forcely calculate distance
        for i in range(len(self.samples)):
            self.samples[i][-1] = self.__distance(self.samples[i], p)
        # sort and find k neighbors 
        ord = list(self.samples.copy())
        ord.sort(key = lambda x: x[-1])
        ord = np.array(ord)
        neighbors = ord[:self.n_neighbors]
        # predict the res
        return self.__output(neighbors)

    def fit(self, X, y):
        assert len(X) == len(y) and len(X) != 0
        # X, y
        self.X = X
        self.y = y
        # get dim
        self.dim = len(X[0])
        self.samples = np.c_[X, y]
        distances = np.zeros(len(self.samples), dtype = float)
        # add one more dim in samples for distance
        self.samples = np.c_[self.samples, distances]
    def predict(self, P):
        res = np.zeros(len(P))
        for i, p in enumerate(P):
            res[i] = self.__predict_point(p)
        return res
    def score(self, X, y):
        '''score without predicted data'''
        y_predicted = self.predict(X)
        res = np.sum(y_predicted == y ) / len(y)
        return res