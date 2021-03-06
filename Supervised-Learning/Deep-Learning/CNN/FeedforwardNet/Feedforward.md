# 前馈神经网络: Feedforward Neural Network

*Author: Limzh*

## 一. 简介

前馈神经网络中，各个神经元从输入层开始，接受前一级输入，并输出到下一级，直到输出层。整个网络中无反馈，可以用一个有向无环图标识。前馈神经网络采用一种单项多层结构。其中每一层包含若干个神经元，同一层的神经元之间没有相互连接，层间信息的传送只沿着一个方向进行。其中第一层成为输入层，最后一层为输出层，中间为隐含层，简称隐层。

对于前馈神经网络的结构设计，通常采用的方法有三类，直接定型法、修剪法和生长法。

## 二. 常见前馈神经网络

1. 感知机是最简单的前馈网络，它主要用于模式分类，也可用在基于模式分类的学习控制中。
2. BP网络是指优化算法调整采用了反向传播学习算法的前馈网络。
3. RBF网络是指隐含层神经元由RBF神经元组成的前馈网络

## 三. 反向传播算法

