#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 16:36:01 2019
图的基本操作
Tensorflow中默认的命名规则：
一般在需要使用名字时，都会在定义的同时为它指定好固定的名字。

@author: zhuyangze
"""

#import numpy as np
import tensorflow as tf

# c是在刚开始的默认图中建立的
c = tf.constant(0.0)

# tf.Graph函数建立一个图
g = tf.Graph()

with g.as_default():
    c1 = tf.constant(0.0)
    print("c1.graph: ", c1.graph)
    
    # 在图里面可以通过名字得到其对应的元素
    print("c1.name: ", c1.name)
    t = g.get_tensor_by_name(name = "Const:0")
    print("t: ", t)
    
    print("g: ", g)
    print("c.graph: ", c.graph)

g2 = tf.get_default_graph()
print("g2: ", g2)

# 重新建立一张图来代替原来的默认图
tf.reset_default_graph()
g3 = tf.get_default_graph()
print("g3: ", g3)

print("获取节点操作: ")

a = tf.constant([[1.0, 2.0]])
b = tf.constant([[1.0], [3.0]])

tensor1 = tf.matmul(a, b, name='exampleop')
print(tensor1.name, tensor1)
test = g3.get_tensor_by_name("exampleop:0")
print("test: ", test)
print("tensor1.op.name: ", tensor1.op.name)

# OP其实是描述张量中的运算关系，是通过访问张量的属性找到的
testop = g3.get_operation_by_name("exampleop")
print("testop: ", testop)

with tf.Session() as sess:
    test = sess.run(test)
    print(test)
    test = tf.get_default_graph().get_tensor_by_name("exampleop:0")
    print(test)

print("获取元素列表:")

# 查看图中的全部元素
tt2 = g.get_operations()
print(tt2)

print("根据名字来获取元素:")
# as_graph_element 传入的是一个对象，返回一个张量或是一个op,具有验证和转换功能
tt3 = g.as_graph_element(c1)
print("tt3", tt3)

# 练习
# get_default_graph放在as_default作用域里
print("练习:")
with g.as_default():
    c1 = tf.constant(0.0)
    print("c1: ", c1.graph)
    print(g)
    print("c: ", c.graph)
    
    g3 = tf.get_default_graph()
    print("g3: ", g3)
    