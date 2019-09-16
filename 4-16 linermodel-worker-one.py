#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 08:59:00 2019
使用TensorFlow实现分布式部署训练
在本机通过3个端口来建立3个终端，分别是一个ps, 两个worker,实现TensorFlow的分布式运算

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

# 定义变量
plotdata = {"batchsize":[], "loss": []}

def moving_averange(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx - w):idx])/w for idx, val in enumerate(a)]

train_X = np.linspace(-1, 1, 100)
# y = 2x,加入了嗓音。
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3

# 显示模拟数据。
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.legend()
plt.show()

tf.reset_default_graph()

# 定义IP和端口
strps_hosts = "localhost:1681"
strworker_hosts="localhost:1682,localhost:1683"

# 定义角色名称: worker
strjob_name = "worker"
task_index = 0

# 将字符串转成数组
ps_hosts = strps_hosts.split(',')
worker_hosts = strworker_hosts.split(',')
cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})

# 创建server
server = tf.train.Server(
        {'ps': ps_hosts, 'worker': worker_hosts},
        job_name = strjob_name,
        task_index = task_index)

# ps角色添加等待函数
if strjob_name == 'ps':
    print("wait")
    # 使用server.join()函数进行线程挂起，开始接收连接消息
    server.join()

with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % task_index,
        cluster=cluster_spec)):
    
    # 创建模型
    # 点位符
    X = tf.placeholder("float")
    Y = tf.placeholder("float")
    
    # 模型参数
    W = tf.Variable(tf.random_normal([1]), name="weight")
    b = tf.Variable(tf.zeros([1]), name="bias")
    
    # 获得迭代次数
    # 为了使载入检查点文件时能够同步循环次数，加了一个global_step变量，并将其放到优化器中。
    # global_step = tf.train.get_or_create_global_step()
    global_step = tf.contrib.framework.get_or_create_global_step()
    
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
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step)
    
    saver = tf.train.Saver(max_to_keep=1)
    # 合并所有summary
    merged_summary_op = tf.summary.merge_all()
    
    # 初始化所有变量
    init = tf.global_variables_initializer()

# 定义参数
# 迭代次数
training_epochs = 100
display_step = 2

# 创建Supervisor,管理session
sv =tf.train.Supervisor(is_chief = (task_index == 0),
                        logdir = "log/super/",
                        init_op = init,
                        summary_op = None,
                        saver = saver,
                        global_step = global_step,
                        save_model_secs = 5)

# 连接目标角色创建session
with sv.managed_session(server.target) as sess:
    print("sess ok")
    print(global_step.eval(session=sess))
    
    for epoch in range(global_step.eval(session=sess), training_epochs * len(train_X)):
        
        for (x, y) in zip(train_X, train_Y):
            _, epoch = sess.run([optimizer, global_step], feed_dict={X: x, Y: y})
            # 生成summary
            summary_str = sess.run(merged_summary_op, feed_dict={X: x, Y: y})
            # 将summary写入文件
            sv.summary_computed(sess, summary_str, global_step=epoch)
            
            # 显示训练中的详细信息
            if epoch % display_step == 0:
                # 通过feed机制将真实数据灌到点位符对应的位置
                loss = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
                print("Epoch:", epoch + 1, "cost=", loss, "W=",sess.run(W), "b=", sess.run(b))
                if not (loss == "NA"):
                    plotdata["batchsize"].append(epoch)
                    plotdata["loss"].append(loss)
        
    print("Finished!")
    sv.saver.save(sess, "log/mnist_with_summaries/" + "sv.cpk", global_step = epoch)

sv.stop()
