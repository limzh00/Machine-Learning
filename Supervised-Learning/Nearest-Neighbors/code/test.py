from tqdm import tqdm 
import numpy as np 
from time import sleep
K = np.zeros(10)
bar = enumerate(tqdm(K))
for i in bar:
    print(i)
    sleep(0.5)
