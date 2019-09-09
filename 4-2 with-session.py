#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 11:33:22 2019

@author: zhuyangze
"""

import tensorflow as tf

# 定义常量
a = tf.constant(3)
b = tf.constant(4)

# 建立session
with tf.Session() as sess:
    print("相加: %i" % sess.run(a + b))
    print("相乘: %i" % sess.run(a * b))