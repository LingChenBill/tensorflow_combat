#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 11:12:49 2019

@author: zhuyangze
"""

import tensorflow as tf

# 定义一个常量
hello = tf.constant('Hello, TensorFlow!')
print(hello)
# 建立一个session
sess = tf.Session()
# 通过session里面run
print(sess.run(hello))
# 关闭session
sess.close