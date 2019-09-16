#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 20:53:23 2019
MNIST数据集
利用Tensorflow代码下载MNIST
MNIST数据集中的图片是28*28 Pixel.
每一幅图就是1行784（28*28）列的数据

MNIST里包含3个数据集:
    第一个是训练数据集
    第二个是测试数据集
    第三个是验证数据集

利用验证数据集可以评估出模型的准确度，
这个准确度越高，代表模型的泛化能力越强。

@author: zhuyangze
"""
from tensorflow.examples.tutorials.mnist import input_data

# one_hot=True，表示将样本标签转化为one_hot编码
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("Download MNIST data completed.")

print('输入数据: ', mnist.train.images)
# 打印出来的是训练集的图片信息，是一个55000行，784列的矩阵
print('输入数据集shape: ', mnist.train.images.shape)

import pylab
im = mnist.train.images[1]
im = im.reshape(-1, 28)

pylab.imshow(im)
pylab.show()

# 测试数据集和验证数据集
print('输入测试数据集shape: ', mnist.test.images.shape)
print('输入验证数据集shape: ', mnist.validation.images.shape)