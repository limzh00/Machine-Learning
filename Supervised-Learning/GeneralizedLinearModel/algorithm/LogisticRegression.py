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
    def fit(self, X, y):
        self.X = X
        self.y = y
    def predict(self, P):
        pass
    def score(self, X, y):
        return 1

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
    