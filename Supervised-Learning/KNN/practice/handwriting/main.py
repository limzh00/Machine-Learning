import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import scipy as sp 
import sys
import cv2
sys.path.append('..')
from keras.datasets import mnist
from module.kdTree import kdTree
from sklearn import neighbors
from datetime import datetime

# self-defined data_processor
from DataProcessor import DataProcessor

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
    clf = neighbors.KNeighborsClassifier(n_neighbors=10, weights='uniform')
    clf.fit(x_train[:], y_train[:])
    # 4. predict
    # -- load custom_data
    data_processor = DataProcessor(load_dir = './dataset1-raw/', save_dir = './dataset1-processed/', img_height = 28, img_width = 28)
    data_processor.data_process(is_all=False)
    (X, y) = data_processor.data_load()
    z = clf.predict(X)
    # output the result
    for i in range(len(y)):
        print(f"The predicted value: {z[i]} \n The true value: {y[i]}")
    accuracy = np.sum(z == y) / len(y)
    print(f"Accuracy: {accuracy}")
    end = datetime.now()
    # print(accuracy)
    print(end - start)

if __name__ == '__main__':
    main()