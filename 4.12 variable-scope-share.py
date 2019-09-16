#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 11:08:27 2019
variable_scope 中 get_variable 初始化的继承功能，以及嵌套variable_scope的继承功能

@author: zhuyangze
"""

import tensorflow as tf

tf.reset_default_graph()

# variable_scope 和 get_variable 都有初始化的功能
# 在初始化时，如果没有对当前变量初始化，
# 则Tensorflow会默认使用作用域的初始化方法对其初始化，
# 并且作用域的初始化方法也有继承功能。
with tf.variable_scope("test1", initializer=tf.constant_initializer(0.4)):
    var1 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)
    
    with tf.variable_scope("test2"):
        var2 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)
        var3 = tf.get_variable("var3", shape=[2], initializer=tf.constant_initializer(0.3))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("var1 = ", var1.eval())
    print("var2 = ", var2.eval())
    print("var3 = ", var3.eval())