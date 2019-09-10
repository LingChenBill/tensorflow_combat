#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 19:55:38 2019
矩阵相关的操作

@author: zhuyangze
"""

import tensorflow as tf
import numpy as np

diagonal = [1, 2, 3, 4]
# 返回一个给定对角线的对角tensor
diag = tf.diag(diagonal)

# 将一个给定对角线的对角tensor,转换成tensor。与tf.diag功能相反
diag_part = tf.diag_part(diag)

# 求一个二维Tensor足迹，即对角值diagonal之和
trace = tf.trace(diag)

# 让输入a按照参数perm指定的维度顺序进行转置操作
t = [[1, 2, 3], [4, 5, 6]]
transpose = tf.transpose(t)
transpose_perm = tf.transpose(t, [1, 0])

# 沿着指定的维度对输入进行反转
s = [[[[0, 1, 2, 3],
       [4, 5, 6, 7],
       [8, 9, 10, 11]],
       [[12, 13, 14, 15],
       [16, 17, 18, 19],
       [20, 21, 22, 23]]]]

# dim为t.shape里的索引
dim = [3]
# 进行反转操作
s_t = tf.reverse(s, dim)
# 按多个轴反转
s_t_others = tf.reverse(s, [1, 2])

with tf.Session() as sess:
    print("返回一个给定对角线的对角tensor:")
    print(sess.run(diag))
    print("将一个给定对角线的对角tensor,转换成tensor:")
    print(sess.run(diag_part))
    print("对角值diagonal之和:")
    print(sess.run(trace))
    print("让输入a按照参数perm指定的维度顺序进行转置操作:")
    print(sess.run(transpose))
    print(sess.run(transpose_perm))
    
    print("矩阵反转:")
    print(np.shape(s))
    print(sess.run(s_t))
    print(sess.run(s_t_others))

# 求解矩阵方程。
# 2x + 3y = 12
# x + y = 5

sess2 = tf.InteractiveSession()
a = tf.constant([[2., 3.], [1., 1.]])
print("求解矩阵方程:")
print(tf.matrix_solve(a, [[12.], [5.]]).eval())
# sess2.close()