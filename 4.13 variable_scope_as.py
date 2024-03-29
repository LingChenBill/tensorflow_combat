#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 11:29:12 2019
演示variable_scope的as用法，以及对应的作用域


@author: zhuyangze
"""
import tensorflow as tf

tf.reset_default_graph()

with tf.variable_scope("scope1") as sp:
    var1 = tf.get_variable("v", [1])

print("sp: ", sp.name)
print("var1: ", var1.name)

with tf.variable_scope("scope2"):
    var2 = tf.get_variable("v", [1])
    
    with tf.variable_scope(sp) as sp1:
        var3 = tf.get_variable("v3", [1])
        
        with tf.variable_scope(""):
            var4 = tf.get_variable("v4", [1])

print("sp1", sp1.name)
print("var2", var2.name)
print("var3", var3.name)

# 操作符不仅受到tf.name_scope作用域的限制，同时也受到tf.name_scope作用域限制
with tf.variable_scope("scope"):
    with tf.name_scope("bar"):
        # tf.name_scope只能限制op,不能限制变量的命名。
        v = tf.get_variable("v", [1])
        x = 1.0 + v
        
        with tf.name_scope(""):
            y = 1.0 + v

print("v: ", v.name)
print("x.op: ", x.op.name)

# 打印variable_scope与name_scope在空字符情况的值
print("var4: ", var4.name)
print("y: ", y.op.name)