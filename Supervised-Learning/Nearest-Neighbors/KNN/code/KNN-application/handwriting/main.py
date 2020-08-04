import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import scipy as sp 
import sys
sys.path.append('..')
from keras.datasets import mnist
from module.kdTree import kdTree
from sklearn import neighbors
from datetime import datetime

def main():
    start = datetime.now()
    # 1. load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # check data info: for each img, 28 x 28 x 1 (cmap = binary)
    # x_train: 60000, x_test: 10000
    (img_height, img_width) = x_train[0].shape
    # 2. preprocessing
    # flatten img
    x_train = x_train.reshape(len(x_train), img_height * img_width)
    x_test  = x_test.reshape(len(x_test), img_height * img_width)
    # 3. classifier 
    clf = neighbors.KNeighborsClassifier(n_neighbors=10, weights='distance', algorithm='kd_tree')
    clf.fit(x_train, y_train)
    # 4. predict
    accuracy = clf.score(x_test[:100], y_test[:100])
    end = datetime.now()
    print(accuracy)
    print(end - start)

if __name__ == '__main__':
    main()