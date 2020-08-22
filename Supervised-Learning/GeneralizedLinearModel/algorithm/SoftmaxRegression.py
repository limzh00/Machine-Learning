import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_classification
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

class SoftmaxRegression(object):
    def __init__(self, param = None):
        self.X = None
        self.y = None
        self.n_labels = None
        self.n_dims = None
        self.n_samples = None
        self.param = param # m*n 
    def fit(self, X, y):
        assert len(X) == len(y) and len(X)
        self.X = X
        self.n_dims = len(X[0])
        self.n_samples = len(X)
        # preprocess y
        self.n_labels = len(set(y))
        self.y = np.zeros(shape = (self.n_samples,  self.n_labels))
        for i, y_i in enumerate(y):
            self.y[i, int(y_i)] = 1
        # preprocess done
        self.param = self.__MiniBacthGD()
        return 
    def __MiniBacthGD(self, alpha = 0.01, batch_size = 1000, n_iterations = 100000):
        param = np.zeros(shape = (self.n_labels, self.n_dims))
        for _ in enumerate(tqdm(range(n_iterations))):
            index = np.random.randint(low = 0, high = len(self.X), size=batch_size)
            X_batch = self.X[index]
            y_batch = self.y[index]
            y_predicted = self.__predict_original_p(X_batch, param)
            for i in range(self.n_labels):
                for j in range(self.n_dims):
                    param[i,j] -=  alpha * np.sum((y_predicted[:,i] - y_batch[:,i]) * X_batch[:,j]) / batch_size
        return param
    def __predict_original_p(self, X, param):
        # X: (n_samples, n_dims), y:(n_samples, n_labels), theta: (n_labels, n_dims)
        res = np.dot(X, param.transpose())
        return res

    def predict(self,X):
        res = np.zeros(len(X))
        P = self.__predict_original_p(X, self.param)
        for i in range(len(X)):
            res[i] = np.argmax(P[i])
        return res
    def score(self, X, y):
        y_predicted = self.predict(X)
        return np.sum(y_predicted == y) / len(y)


def main():
    (X, y) = make_classification(n_samples=100000,n_features=4,n_classes=4,n_clusters_per_class=1)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    clf = SoftmaxRegression()
    clf.fit(x_train, y_train)
    accur = clf.score(x_test, y_test)
    print(accur)

if __name__ == '__main__':
    main()