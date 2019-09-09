#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 16:27:09 2019
tf.train.Saver函数里面还可以放参数。
将生成的模型里的内容打印出来

@author: zhuyangze
"""

import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

savedir = "log/"
print_tensors_in_checkpoint_file(savedir + "linermodel.cpkt", None, True)

W = tf.Variable(1.0, name="weight")
b = tf.Variable(2.0, name="bias")

# 变量放到一个字典中
saver = tf.train.Saver({'weight': b, 'bias': W})

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver.save(sess, savedir + "linermodel.cpkt")

print_tensors_in_checkpoint_file(savedir + "linermodel.cpkt", None, True)