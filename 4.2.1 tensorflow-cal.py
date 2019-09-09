#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 22:06:19 2019
tensorflow运算

@author: zhuyangze
"""

import numpy as np
import tensorflow as tf

x = tf.constant(2)
y = tf.constant(3)

def f1():
    return tf.multiply(x, 17)

def f2():
    return tf.add(y, 23)

r = tf.cond(tf.less(x, y), f1, f2)
print(r)

# 生成2行3列的张量，值为1
with tf.Session() as sess:
    print(sess.run(r))