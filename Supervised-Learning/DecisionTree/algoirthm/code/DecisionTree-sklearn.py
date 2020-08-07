import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn import tree, datasets
from sklearn.model_selection import train_test_split

def preprocess(total_info:dict) -> tuple:
    return (total_info['data'], total_info['target'], total_info['target_names'])

def main():
    # 1. load data
    (X, y, labels_name) = preprocess(datasets.load_iris())
    # 2. split X and y -- construct test cases and train cases
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    # 3. construct decision tree and fit the data
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf.fit(x_train, y_train)
    # 4. predict and score
    accuracy = clf.score(x_test, y_test)
    # 5. draw this tree
    plt.figure(figsize=(10,10))
    plt.title('sklearn-DecisionTree')
    tree.plot_tree(clf)
    plt.savefig('../img/fig1.png')
    print(accuracy)
    
if __name__ == '__main__':
    main()