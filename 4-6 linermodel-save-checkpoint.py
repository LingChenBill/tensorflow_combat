#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 08:59:00 2019

@author: zhuyangze
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

train_X = np.linspace(-1, 1, 100)
# y = 2x,加入了嗓音。
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3

# 显示模拟数据。
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.legend()
plt.show()

# 重围图
tf.reset_default_graph()

# 定义变量
plotdata = {"batchsize":[], "loss": []}

def moving_averange(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx - w):idx])/w for idx, val in enumerate(a)]

# 创建模型
# 点位符
X = tf.placeholder("float")
Y = tf.placeholder("float")

# 模型参数
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")

# 前向结构
z = tf.multiply(X, W) + b

# 反向优化

# 生成值与真实值的平方差
cost = tf.reduce_mean(tf.square(Y - z))
# 定义一个学习率
learning_rate = 0.01
# 梯度下降
# GradientDescentOptimizer: 一个封装好的梯度下降算法
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 初始化所有变量
init = tf.global_variables_initializer()
# 定义参数
# 迭代次数
training_epochs = 20
display_step = 2

# 生成saver
saver = tf.train.Saver(max_to_keep=1)
savedir = "log/"

# 生成saver
saver = tf.train.Saver()
savedir = "log/"

# 启动session
with tf.Session() as sess:
    sess.run(init)
    # 存放批次值和损失值
    # plotdata = {"batchsize": [], "loss": []}
    # 向模型输入数据
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        
        # 显示训练中的详细信息
        if epoch % display_step == 0:
            # 通过feed机制将真实数据灌到点位符对应的位置
            loss = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print("Epoch:", epoch + 1, "cost=", loss, "W=",sess.run(W), "b=", sess.run(b))
            if not (loss == "NA"):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)
            
            # 保存检查点
            saver.save(sess, savedir + "linermodel.cpkt", global_step=epoch)
    
    print("Finished!")
    
    # 保存模型
    saver.save(sess, savedir + "linermodel.cpkt")
    
    print("cost=", sess.run(cost, feed_dict={X: train_X, Y: train_Y}), "W=", sess.run(W), "b=", sess.run(b))
    
    # 显示模型
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fittedline')
    plt.legend()
    plt.show()
    
    plotdata["avgloss"] = moving_averange(plotdata["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss')
    plt.show()
    
# 重启一个Session, 载入检查点
load_epoch = 18
with tf.Session() as sess2:
    sess2.run(tf.global_variables_initializer())
    saver.restore(sess2, savedir + "linermodel.cpkt-" + str(load_epoch))
    print("x = 0.2, z = ", sess2.run(z, feed_dict={X: 0.2}))

# 快速获取到检查点
with tf.Session() as sess3:
    sess3.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(savedir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess3, ckpt.model_checkpoint_path)
        print("x = 0.2, z = ", sess3.run(z, feed_dict={X: 0.2}))

# 简洁获取检查点
with tf.Session() as sess4:
    kpt = tf.train.latest_checkpoint(savedir)
    if kpt != None:
        saver.restore(sess4, kpt)
        print("x = 0.2, z = ", sess4.run(z, feed_dict={X: 0.2}))




