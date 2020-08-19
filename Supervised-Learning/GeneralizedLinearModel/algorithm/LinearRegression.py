import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import samples_generator
from tqdm import tqdm

class LinearRegressionModel(object):
    def __init__(self, param = None, normalized = False):
        self.param = param
        self.normalized = normalized
        self.X = None
        self.y = None
    def __SGD(self, n_iterations = 100000, alpha = 0.05, batch_size = 10):
        self.param = np.zeros(len(self.X[0]))
        # batch_size = len(self.X) // 100
        for _ in tqdm(range(n_iterations)):
            index = np.random.randint(low = 0, high = len(self.X), size= batch_size)
            X_batch = self.X[index]
            y_batch = self.y[index]
            # tmp predicted values
            y_tmp_predicted = self.predict(X_batch)
            for i in range(len(self.param)):
                self.param[i] -= alpha * np.sum((y_tmp_predicted - y_batch) * X_batch[:,i])            
        return self.param
    def __loss(self, X, y, param):
        y_predicted = np.dot(X, param)
        return np.sum((y - y_predicted)**2) / (2*len(y))
    def fit(self, X, y):
        self.X = X
        self.y = y
        assert len(X) == len(y) and len(X)
        self.param = self.__SGD()
        print("***********TRAINING FINISHED***********")
    def predict(self, X):
        return np.dot(X, self.param)
    def score(self, X, y):
        return self.__loss(X, y, self.param)

def main():
    # generate samples
    (X, y) = samples_generator.make_regression(n_samples=10000, n_features=4)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    clf_sklearn = LinearRegression()
    clf_sklearn.fit(x_train, y_train)
    clf = LinearRegressionModel()
    clf.fit(x_train, y_train)
    print(clf.score(x_test, y_test))
    print(clf_sklearn.score(x_test, y_test))
    
if __name__ == '__main__':
    main()