#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 21:26:56 2019
get_Variable和Variable的区别

@author: zhuyangze
"""

import tensorflow as tf

# 清空图栈
tf.reset_default_graph()

var1 = tf.Variable(1.0, name = 'firstvar')
print("var1:", var1.name)
var1 = tf.Variable(2.0, name = 'firstvar')
print("var1:", var1.name)

var2 = tf.Variable(3.0)
print("var2:", var2.name)
var2 = tf.Variable(4.0)
print("var2:", var2.name)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("var1 = ", var1.eval())
    print("var2 = ", var2.eval())

# get_variable
# 使用get_variable只能定义一次指定名称的变量
get_var1 = tf.get_variable("firstvar", [1], initializer=tf.constant_initializer(0.3))
print("get_var1: ", get_var1.name)
get_var1 = tf.get_variable("firstvar1", [1], initializer=tf.constant_initializer(0.4))
print("get_var1: ", get_var1.name)

with tf.Session() as sess2:
    sess2.run(tf.global_variables_initializer())
    print("get_var1 = ", get_var1.eval())

# get_variable配合variable_scope
with tf.variable_scope("test1", ):
    # 定义作用域, var1和var2都使用firstvar的名字来定义。
    var1 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)

with tf.variable_scope("test2", ):
    var2 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)

print("var1: ", var1.name)
print("var2: ", var2.name)

with tf.Session() as sess3:
    sess3.run(tf.global_variables_initializer())
    print("variable_scope var1 = ", var1.eval())
    print("variable_scope var1 = ", var2.eval())