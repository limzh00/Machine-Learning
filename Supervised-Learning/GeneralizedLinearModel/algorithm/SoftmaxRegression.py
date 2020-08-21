import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_classification
from sklearn.linear_model import LinearRegression

class SoftmaxRegression(object):
    def __init__(self, param = None):
        self.X = None
        self.y = None
        self.param = param
    def fit(self, X, y):
        assert len(X) == len(y) and len(X)
        self.X = X
        self.y = y

    def predict(self,P):
        pass
    def __predict_point(self, p):
        pass
    def score(self):
        pass


def main():
    (X, y) = make_classification(n_samples=100000,n_features=4,n_classes=4)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    clf = SoftmaxRegression()
    clf.fit(x_train, y_train)
    accur = clf.score(x_test, y_test)
    print(accur)

if __name__ == '__main__':
    main()