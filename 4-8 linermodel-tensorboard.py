#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 08:59:00 2019
线性回归可视化
tensorboard --logdir /Users/xxxxxxx/Documents/fork/tensorflow_combat/log/mnist_with_summaries
访问:
http://127.0.0.1:6006/

浏览器最好要用chrome.
在命令行里启动TensorBoard时，一定要先进入日志所在的上级路径下，否则打开的页面里找不到创建好的信息。

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

# 将预测值以直方图显示
tf.summary.histogram('z', z)

# 反向优化

# 生成值与真实值的平方差
cost = tf.reduce_mean(tf.square(Y - z))

# 将损失以标量显示
tf.summary.scalar('loss_function', cost)

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

# 启动session
with tf.Session() as sess:
    sess.run(init)
    
    # 合并所有summary
    merged_summary_op = tf.summary.merge_all()
    # 创建summary_writer
    summary_writer = tf.summary.FileWriter('log/mnist_with_summaries', sess.graph)
    
    # 存放批次值和损失值
    # plotdata = {"batchsize": [], "loss": []}
    # 向模型输入数据
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
            
            # 生成summary
            summary_str = sess.run(merged_summary_op, feed_dict={X: x, Y: y})
            summary_writer.add_summary(summary_str, epoch)
        
        # 显示训练中的详细信息
        if epoch % display_step == 0:
            # 通过feed机制将真实数据灌到点位符对应的位置
            loss = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print("Epoch:", epoch + 1, "cost=", loss, "W=",sess.run(W), "b=", sess.run(b))
            if not (loss == "NA"):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)
    
    print("Finished!")
    print("cost=", sess.run(cost, feed_dict={X: train_X, Y: train_Y}), "W=", sess.run(W), "b=", sess.run(b))
    
    # 图形显示
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
    
    print("x = 0.2, z =", sess.run(z, feed_dict={X: 0.2}))
