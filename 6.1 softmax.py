#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 21:22:34 2019
交叉熵实验
softmax分类问题

@author: zhuyangze
"""

import tensorflow as tf

labels = [[0, 0, 1], [0, 1, 0]]
logits = [[2, 0.5, 6], [0.1, 0, 3]]

logits_scaled = tf.nn.softmax(logits)
logits_scaled2 = tf.nn.softmax(logits_scaled)

# 传入tf.nn.softmax_cross_entropy_with_logits的logits是不需要进行softmax的。
result1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
result2 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits_scaled)

# 自己写loss函数，相当于tf.nn.softmax_cross_entropy_with_logits函数
result3 = -tf.reduce_sum(labels * tf.log(logits_scaled), 1)

with tf.Session() as sess:
    print("scaled = ", sess.run(logits_scaled))
    print("scaled2 = ", sess.run(logits_scaled2))
    
    # 经过第二次的softmax后，分布概率会有变化
    print("rel1 = ", sess.run(result1), "\n")
    print("rel2 = ", sess.run(result2), "\n")
    # 如果将softmax变换完的值放进去后，就相当于第二次softmax的loss,所有会出错
    print("rel3 = ", sess.run(result3))

print("one_hot实验: ")
# 标签总概率为1
labels = [[0.4, 0.1, 0.5], [0.3, 0.6, 0.1]]
result4 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

with tf.Session() as sess:
    # 对于正确分类的交叉熵和错误分类的交叉熵，二者的结果差别没有标准的one_hot那么明显
    print("rel4 = ", sess.run(result4), "\n")

print("sparse交叉熵的使用:")

# sparse 标签
# 表明labels中总共分为3个类: 0, 1, 2
labels = [2, 1]

# sparse_softmax_cross_entropy_with_logits 需要使用非one_hot的标签
result5 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
with tf.Session() as sess:
    print("rel5 = ", sess.run(result5), "\n")

print("计算loss值:")

#注意！！！这个函数的返回值并不是一个数，而是一个向量，
#如果要求交叉熵loss，我们要对向量求均值，
#就是对向量再做一步tf.reduce_mean操作
loss = tf.reduce_sum(result1)
with tf.Session() as sess:
    print("loss = ", sess.run(loss))

labels = [[0 ,0 ,1], [0, 1, 0]]
loss2 = -tf.reduce_sum(labels * tf.log(logits_scaled))
with tf.Session() as sess:
    print("loss2 = ", sess.run(loss2))
