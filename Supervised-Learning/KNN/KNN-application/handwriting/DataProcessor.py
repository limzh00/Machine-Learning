import numpy as np 
import matplotlib.pyplot as plt 
import cv2
import sys
import os
from tqdm import tqdm

class DataProcessor(object):

    def __init__(self, load_dir:str = './dataset1-raw/', save_dir:str = './dataset1-processed/', img_width = 28, img_height = 28):
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.img_height = img_height
        self.img_width = img_width
        # store preprocessed data: numpy.ndarray type
        self.datasets = None
        self.pathes = []
        self.y = []

    def __get_data(self):
        '''get .png / .jpg in given file (not in its subfiles)'''
        for file in os.listdir(self.load_dir):
            if file.endswith('.png') or file.endswith('.jpg'):
                self.pathes.append(self.load_dir + file)
                # For simplicity, the first char of the file name MUST be its truth-label!!!!! 
                self.y.append(int(file[0]))
        # note: all pathes of data are appended into 'self.pathes'
        # get the size of data
        self.datasets = [0] * len(self.pathes)

    def __get_data_all(self):
        '''load all .png / .jpg in given file and its subfiles.'''
        for root, dirs, files in os.walk(self.load_dir):
            for file in files:
                if file.endswith('.png') or file.endswith('.jpg'):
                    self.pathes.append(root + file)
        # note: all pathes of data are appended into 'self.pathes'
        # get the size of data
        self.datasets = np.zeros(shape = len(self.pathes), dtype=object) # dtype is important here
    
    def data_process(self, is_all:bool = False):
        # get data pathes:
        if not is_all: self.__get_data()
        else: self.__get_data_all()
        # for each data file, read with cv2.
        print("*********************PROCESSING DATA******************************")
        for i, image_addr in enumerate(tqdm(self.pathes)):
            image = cv2.imread(image_addr, 0) # read as gray image (np.array type)
            # 1. little technique: linear transformation --> enhance contrast degree
            image = float(2) * image # linear transformation
            image[image > 255] = 255  # if one is beyond 255, set 255
            # image = k * image, if k is float, we should round all values.
            # Here, even though we set k with value 2, for generality, we still do so.
            image = (np.round(image)).astype(np.uint8)
            # 
            retval, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

            # 2. resize image
            # note: the third param is just an algorithm to do interpolation.
            # Dont bother, it counts little.
            image = cv2.resize(image, (self.img_height, self.img_width), interpolation = cv2.INTER_LINEAR)

            # 3. save the preproccessed data
            # note: [int(cv2.IMWRITE_PNG_COMPRESSION, 0)] represents the level of compression when img saves. Levels ranges from 0 to 9. The default value is 3, here we set it as value 0 (no quality loss).
            cv2.imwrite(f"{self.save_dir}fig{i}.png", image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

            # 4. update self.pathes to track images
            self.pathes[i] = self.save_dir + f"fig{i}.png"
        print("**************************DONE************************************")

    def data_load(self) -> tuple:
        # 1. load data
        print("*********************LOADING DATA*********************************")
        for i, image_addr in enumerate(tqdm(self.pathes)):
            image = cv2.imread(image_addr, 0) # grayscale image
            # each input is required to be a vector of 1 x 784
            image = image.ravel() # just flatten it is OK. Or, image = image.reshape(1, 28*28)
            # load image in self.datasets, and done.
            self.datasets[i] = image 
        # after loading, we should maintain self.datasets as a np.ndarray
        self.datasets = np.array(self.datasets)    
        print("**************************DONE************************************")     
        
        # 2. return the result
        return (self.datasets, np.array(self.y))