# 机器学习实践记录：从KNN算法到KD树

*Author: Limzh*

## Section1. 序言

在模式识别中，k近邻算法是一个非参数统计学的的分类算法。它的基本原理是，

> 给定一个数据集$P = \{p_1, p_2,...,p_{m-1}\}$，$p_i \in \mathbb{R}^n, P \in \mathbb{R}^m\times\mathbb{R}^n$，以及一个相应的标签集 $L = \{l_1,l_2,...,l_{m-1}\}, l_i \in \{0,1,...,q\}$，q为标签集中的类别数。输入一预测点 $p \in \mathbb{R}^n$, 找到 数据集中离p最近的k个点，并依据它们的标签值给出预测点的标签值，并输出。

简而言之，KNN算法秉持着“物以类聚，人以群分”的自然产生的思想，不寻找数据的分布模式（这也是非参数的原因，无模型），而是通过数据集和预测点的距离关系输出预测点的标签。

通过调用`sklearn`中的相关函数，我们可以直接窥得KNN算法的实现结果。

## sklearn 中 KNN 算法

### 1.1 导入库介绍

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

### 1.2 生成数据学习

```python
X, y = make_classification(n_)
```

## 附录

### 1. [sklearn库的结构](https://blog.csdn.net/algorithmPro/article/details/103045824?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.channel_param)

