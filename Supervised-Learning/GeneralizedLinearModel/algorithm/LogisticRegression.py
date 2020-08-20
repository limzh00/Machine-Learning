import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import samples_generator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

class LogisticRgression_simple(object):
    def __init__(self):
        self.X = None
        self.y = None
        self.param = None
    def __MiniBatchGD(self, batch_size = 10, n_iterations = 10000, alpha = 0.1):
        self.param = np.zeros(len(self.X[0]))
        # batch_size
        for _ in tqdm(range(n_iterations)):
            index = np.random.randint(low = 0, high = len(self.X), size = batch_size)
            X_batch = self.X[index]
            y_bacth = self.y[index]
            # tmp predicted values
            y_tmp_predicted = self.__predict_original_p(X_batch)
            # update param for one iteration
            for i in range(len(self.param)):
                self.param[i] -= alpha * np.sum((y_tmp_predicted - y_bacth) * X_batch[:,i])
            # print(self.__loss(self.X, self.y))
        return self.param
    def __predict_point(self, p):
        eta = np.dot(self.param, p)
        p = 1 / (1 + np.exp(-eta))
        return p
    def __loss(self, X, y):
        y_predicted = self.__predict_original_p(X)
        loss = - np.dot(np.log(y_predicted), y) - np.dot((1 - y), np.log(1 - y_predicted))
        return loss
    def fit(self, X, y):
        self.X = X
        self.y = y
        assert len(X) == len(y) and len(X)
        self.param = self.__MiniBatchGD()
        print("**********TRAINING FINISHED**********")
    def __predict_original_p(self, P):
        P = np.array(P)
        res = np.zeros(len(P))
        for i, p in enumerate(P):
            res[i] = self.__predict_point(p)
        return res
    def predict(self, P):
        res = self.__predict_original_p(P)
        res[res >= 0.5] = 1
        res[res < 0.5] = 0
        return res
    def score(self, X, y):
        res = self.predict(X)
        return np.sum(res == y) / len(y)

def main():
    X, y = samples_generator.make_classification(n_samples = 10000, n_features = 4)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    clf_sklearn = LogisticRegression()
    clf = LogisticRgression_simple()
    # fit
    clf_sklearn.fit(x_train, y_train)
    clf.fit(x_train, y_train)
    # score
    print(clf_sklearn.score(x_test, y_test))
    print(clf.score(x_test, y_test))

if __name__ == '__main__':
    main()
    