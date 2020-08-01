# 机器学习实践记录：从KNN算法到KD树

*Author: Limzh*

## Section1. 序言

在模式识别中，k近邻算法是一个非参数统计学的的分类算法。它的基本原理是，

> 给定一个数据集$P = \{p_1, p_2,...,p_{m-1}\}$，$p_i \in \mathbb{R}^n, P \in \mathbb{R}^m\times\mathbb{R}^n$，以及一个相应的标签集 $L = \{l_1,l_2,...,l_{m-1}\}, l_i \in \{0,1,...,q\}$，q为标签集中的类别数。输入一预测点 $p \in \mathbb{R}^n$, 找到 数据集中离p最近的k个点，并依据它们的标签值给出预测点的标签值，并输出。

简而言之，KNN算法秉持着“物以类聚，人以群分”的自然产生的思想，不寻找数据的分布模式（这也是非参数的原因，无模型），而是通过数据集和预测点的距离关系输出预测点的标签。

官方文档的前言如下

> 最近邻方法背后的原理是从训练样本中找到与新点在距离上最近的预定数量的几个点，然后从这些点中预测标签。 这些点的数量可以是用户自定义的常量（K-最近邻学习）， 也可以根据不同的点的局部密度（基于半径的最近邻学习）确定。距离通常可以通过任何度量来衡量： standard Euclidean distance（标准欧式距离）是最常见的选择。Neighbors-based（基于邻居的）方法被称为 *非泛化* 机器学习方法， 因为它们只是简单地”记住”了其所有的训练数据（可能转换为一个快速索引结构，如 [Ball Tree](https://sklearn.apachecn.org/docs/master/7.html#1643-ball-树) 或 [KD Tree](https://sklearn.apachecn.org/docs/master/7.html#1642-k-d-树)）。
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

numpy，pandas，scipy 和 matplotlib 是数据处理和可视化的标准库.
sklearn是python的机器学习算法库之一，全称scikit-learn。它集成了四大类机器学习算法，包括分类，回归，降维和聚类。同时它还支持生成算法需要的标数据集，数据的预处理以及数据的引入等功能。`make_classification`就是一种生成样本数据的数据生成器。`neighbors`包装了大量最近邻有关算法。

### 2.2 生成数据

```python
X, y = make_classification(n_samples=200, n_features=2 ,n_redundant=0, n_clusters_per_class=1, n_classes=3)
```

上面一行代码生成200个数据样例，每个样本有两个特征。标签值的取值范围为0，1，2，意味着所有点被分成三类。

### 2.3 创建待预测输入

```python

```



## 附录

### 1. [sklearn库的结构](https://blog.csdn.net/algorithmPro/article/details/103045824?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.channel_param)

