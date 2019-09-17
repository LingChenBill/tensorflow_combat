#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 20:53:23 2019
MNIST数据集
1) 导入MNIST数据集
2）分析MNIST样本特点定义变量
3）构建模型
4）训练模型并输出中间状态参数
5）测试模型
6）保存模型
7）读取模型

使用sparse_softmax_cross_entropy_with_logits函数来运算交叉熵

@author: zhuyangze
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")
# import pylab

# tf.compat.v1.disable_eager_execution()
tf.reset_default_graph()

# 定义占位符
# MNIST数据集的维度是28*28=784, 0~9 数字，分类10
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.int32, [None])

# 定义学习参数，用于计算输入值
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 构建模型
z = tf.matmul(x, W) + b
# Softmax分类
pred = tf.nn.softmax(z)

# 损失函数
# 将生成的pred与样本标签y进行一次交叉熵的运算，然后取平均值
# cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices = 1))
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=z))

# 定义参数
learning_rate = 0.01

# 使用梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# training_epochs代表要把整个训练样本集迭代25次
training_epochs = 25
# batch_size代表在训练过程中一次取出100条数据进行训练
batch_size = 100
# display_step代表每训练一次就把具体的中间状态显示出来
display_step = 1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # 遍历全部数据集
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        
        # 显示训练中详细信息
        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "Cost=", "{:.9f}".format(avg_cost))
    
    print("Finished!")