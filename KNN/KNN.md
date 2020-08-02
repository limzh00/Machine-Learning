# 机器学习实践记录：从KNN算法到KD树

*Author: Limzh*

## Section1. 序言

在模式识别中，k近邻算法是一个非参数统计学的的分类算法。KNN算法秉持着“物以类聚，人以群分”的自然产生的思想，不寻找数据的分布模式（这也是非参数的原因，无模型），而是通过数据集和预测点的距离关系输出预测点的标签。

官方文档的前言如下

> 最近邻方法背后的原理是从训练样本中找到与新点在距离上最近的预定数量的几个点，然后从这些点中预测标签。 这些点的数量可以是用户自定义的常量（K-最近邻学习）， 也可以根据不同的点的局部密度（基于半径的最近邻学习）确定。距离通常可以通过任何度量来衡量.Neighbors-based（基于邻居的）方法被称为 *非泛化* 机器学习方法， 因为它们只是简单地”记住”了其所有的训练数据（可能转换为一个快速索引结构，如 [Ball Tree](https://sklearn.apachecn.org/docs/master/7.html#1643-ball-树) 或 [KD Tree](https://sklearn.apachecn.org/docs/master/7.html#1642-k-d-树)）。
>
> 尽管它简单，但最近邻算法已经成功地适用于很多的分类和回归问题，例如手写数字或卫星图像的场景。 作为一个 non-parametric（非参数化）方法，它经常成功地应用于决策边界非常不规则的分类情景下。

通过调用`sklearn`中的相关函数，我们可以直接窥得KNN算法的实现结果。

## Section2. sklearn 中 KNN 算法

### 2.1 导入库介绍

```python
import numpy as np
import pandas as pandas
import scipy
import matplotlib.pyplt as plt 
from matplotlib.colors import ListedColormap
from scipy import stats
from sklearn import neighbors, datasets
from sklearn.datasets.samples_generator import make_classification 
```

numpy，pandas，scipy 和 matplotlib 是数据处理和可视化的标准库. sklearn是python的机器学习算法库之一，全称scikit-learn。它集成了四大类机器学习算法，包括分类，回归，降维和聚类。同时它还支持生成算法需要的标数据集，数据的预处理以及数据的引入等功能。`make_classification`是数据生成器。`neighbors`包装了大量最近邻有关算法.

### 2.2 生成数据

```python
X, y = make_classification(n_samples=200, n_features=2 ,n_redundant=0, n_clusters_per_class=1, n_classes=3)
```

上面一行代码生成200个数据样例，每个样本有两个特征。标签值的取值范围为0，1，2，意味着所有点被分成三类。

### 2.3 创建待预测输入

```python
# 2. generate to-be-classified data (predicted data)
# -- get the range of to-be-classified point 
x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
# -- mesh 
h = .01 # step size for mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# -- flatten and pair up
P = np.c_[xx.ravel(),yy.ravel()] # P is an vector of points 
```

由于每个样本只具有两个特征，因此整个样本空间为二维平面，由于离样本集合太远的点没有太大的预测价值，因此我们仅关注在样本集合附近的点即可。故而，我们用`x_min, x_max, y_min, y_max`将样本集合所在的矩形圈定起来，在该矩形内的所有数据点构成的集合就是我们要预测的样本数据。在这里，我们使用`np.mesh()`函数生成网格点.（xx和yy的具体数值请自己尝试输出） 最后，得到P作为所有预测点的集合。P是一个一维数组，每一个元素是平面一点的坐标。至此，待预测数据已然完成。

### 2.4 导入分类器并拟合

```python
 # 3. generate classifier
 clf = neighbors.KNeighborsClassifier(n_neighbors=15, weights='distance')
 # 4. fit clf with trained-data
 clf.fit(X, y)
```

### 2.5 预测并将图绘制

```python

# 5. predict P with fitted classifier and get the result Z full of prediected value
Z = clf.predict(P) # note that it requires P is a vector of 1 dimension

# 6. plot the result。
# -- for scattered points (trained data)
color_bold = ListedColormap(['#156589', '#199934', '#F9AB3B'])

# -- for the predicted plane (to-be-classified data)
color_light = ListedColormap(['#B2EBF2','#DCEDC8','#FFE0B2'])


# note that the backgroud(color_light should be printed before to avoid overlapping)
plt.pcolormesh(xx, yy, Z.reshape(xx.shape), cmap = color_light)
plt.scatter(X[:,0], X[:,1], c=y, cmap=color_bold)


# add title
plt.title('sklearn-15NN')
plt.show()




```

### 2.6 完整代码

```python
#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Minzhang Li
# FILE: F:\MyGithubs\Machine-Learning\Supervised-Learning\Nearest-Neighbors\code\KNN_sklearn.py
# DATE: 2020/08/01 Sat
# TIME: 11:01:58

# DESCRIPTION: This file offers an example demonstrating how sklearn lib works.

import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from sklearn.datasets import  make_classification
from sklearn import  neighbors, datasets
from tqdm import tqdm

def main():
    # 1. generate samples. 200 samples for which are of two dims, 3-classes labels. note that dont mess up 'class' with 'cluster'
    X, y = make_classification(n_samples=200, n_features=2 ,n_redundant=0, n_clusters_per_class=1, n_classes=3)
    # 2. generate to-be-classified data (predicted data)
    # -- get the range of to-be-classified point 
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    # -- mesh 
    h = .01 # step size for mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # -- flatten and pair up
    P = np.c_[xx.ravel(),yy.ravel()] # P is an vector of points 
    # 3. generate classifier
    clf = neighbors.KNeighborsClassifier(n_neighbors=15, weights='distance')
    # 4. fit clf with trained-data
    clf.fit(X, y)
    # 5. predict P with fitted classifier and get the result Z full of prediected value
    Z = clf.predict(P) # note that it requires P is a vector of 1 dimension
    # 6. plot the result.
    # -- for scattered points (trained data)
    color_bold = ListedColormap(['#156589', '#199934', '#F9AB3B'])
    # -- for the predicted plane (to-be-classified data)
    color_light = ListedColormap(['#B2EBF2','#DCEDC8','#FFE0B2'])
    # note that the backgroud(color_light should be printed before to avoid overlapping)
    plt.pcolormesh(xx, yy, Z.reshape(xx.shape), cmap = color_light)
    plt.scatter(X[:,0], X[:,1], c=y, cmap=color_bold)
    # add title
    plt.title('sklearn-15NN')
    plt.show()

    
if __name__ == '__main__':
    main() 
```

### 2.6 结果

<img src="F:\MyGithubs\Machine-Learning\Supervised-Learning\Nearest-Neighbors\img\sklearn_1.png" style="zoom: 33%;" />



## 附录

### 1. [sklearn库的结构](https://blog.csdn.net/algorithmPro/article/details/103045824?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.channel_param)

