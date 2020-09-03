import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import scipy as sp 
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical

def download_mnist():
    train_X, train_y = mnist.load_data()[0]
    train_X = train_X.reshape(-1, 28, 28, 1)
    train_X = train_X.astype('float32')
    train_X = train_X / 255
    train_y = to_categorical(train_y, 10)

download_mnist()







