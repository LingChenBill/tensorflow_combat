#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 20:47:06 2019

@author: zhuyangze
"""

import tensorflow as tf

tf.reset_default_graph()
global_step = tf.train.get_or_create_global_step()
step = tf.assign_add(global_step, 1)

# 设置检查点路径为log/checkpoints
with tf.train.MonitoredTrainingSession(checkpoint_dir = 'log/checkpoints', save_checkpoint_secs = 2) as sess:
    print(sess.run([global_step]))
    # 启用死循环，当sess不结束时就输出
    while not sess.should_stop():
        i = sess.run(step)
        print(i)