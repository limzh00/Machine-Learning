import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import math

class Node(object):
    def __init__(self, l_node = None, r_node = None, feature = None, father = None, r = None):
        self.left_node = l_node
        self.right_node = r_node
        self.feature = feature
        self.father = father
        # self.r is the dimension on which we split the array. 
        self.r = r

class KD(object):
    def __init__(self, samples):
        self.root = Node(,)
    def 
