#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 20:53:23 2019
MNIST数据集
1) 导入MNIST数据集
2）分析MNIST样本特点定义变量
3）构建模型
4）训练模型并输出中间状态参数
5）测试模型
6）保存模型
7）读取模型


@author: zhuyangze
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import pylab

tf.compat.v1.disable_eager_execution()
tf.reset_default_graph()

# 定义占位符
# MNIST数据集的维度是28*28=784, 0~9 数字，分类10
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 定义学习参数，用于计算输入值
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 构建模型
# Softmax分类
pred = tf.nn.softmax(tf.matmul(x, W) + b)

# 损失函数
# 将生成的pred与样本标签y进行一次交叉熵的运算，然后取平均值
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices = 1))

# 定义参数
learning_rate = 0.01

# 使用梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# training_epochs代表要把整个训练样本集迭代25次
training_epochs = 25
# batch_size代表在训练过程中一次取出100条数据进行训练
batch_size = 100
# display_step代表每训练一次就把具体的中间状态显示出来
display_step = 1
# 保存模型
saver = tf.train.Saver()
model_path = "log/mnist_model.ckpt"


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        
        # 循环所有数据集
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 运行优化器
            _, c = sess.run([optimizer, cost], feed_dict = {x: batch_xs, y: batch_ys})
            # 计算平均loss值
            avg_cost += c / total_batch
        
        # 显示训练中详细信息
        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "Cost=", "{:.9f}".format(avg_cost))
    
    print("Finished!")
    
    # 测试Model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    
    # 保存模型
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)
    
# 读取模型
print("Starting 2nd session....")
with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())
    # 恢复模型变量
    saver.restore(sess, model_path)
    
    # 测试model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    
    output = tf.argmax(pred, 1)
    batch_xs, batch_ys = mnist.train.next_batch(2)
    outputval, predv = sess.run([output, pred], feed_dict = {x: batch_xs})
    # 第一个数组：输出是预测结果
    # 第二个数组：输出预测出来的真实输出值
    # 第三个数组：是标签值oneshot编码表示的值
    print(outputval, predv, batch_ys)
    
    im = batch_xs[0]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()

    im = batch_xs[1]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()





