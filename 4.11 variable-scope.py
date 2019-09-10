#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 22:17:37 2019
1) variable_scope还支持嵌套
2) 共享变量功能的实现

@author: zhuyangze
"""

import tensorflow as tf

# 将图(一个计算任务)里面的变量清空
tf.reset_default_graph()

with tf.variable_scope("test1", ):
    var1 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)
    
    with tf.variable_scope("test2", ):
        var2 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)

print("var1: ", var1.name)
print("var2: ", var2.name)


# 共享变量:reuse=tf.AUTO_REUSE
with tf.variable_scope("test1", reuse=tf.AUTO_REUSE):
    var3 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)
    
    with tf.variable_scope("test2"):
        var4 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)

print("var3: ", var3.name)
print("var4: ", var4.name)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("var1 value: ", var1.eval())
    print("var2 value: ", var2.eval())
    print("var3 value: ", var3.eval())
    print("var4 value: ", var4.eval())