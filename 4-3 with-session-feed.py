#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 11:40:30 2019

@author: zhuyangze
"""

import tensorflow as tf

# 定义占位符
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a, b)
mul = tf.multiply(a, b)

with tf.Session() as sess:
    # 使用feed机制将具体数值通过占位符传入
    print("相加: %i" % sess.run(add, feed_dict={a: 3, b: 4}))
    print("相乘: %i" % sess.run(mul, feed_dict={a: 3, b: 4}))
    print(sess.run([add, mul], feed_dict={a:3, b: 4}))