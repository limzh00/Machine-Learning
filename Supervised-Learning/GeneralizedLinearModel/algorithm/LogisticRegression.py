import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import samples_generator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

def main():
    X, y = samples_generator.make_classification(n_samples = 10000, n_features = 4)
    x_train, y_train, x_test, y_test = train_test_split(X, y, test_size = 0.2)
    