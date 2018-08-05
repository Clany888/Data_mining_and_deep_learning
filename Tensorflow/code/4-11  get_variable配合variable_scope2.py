# -*- coding: utf-8 -*-
import tensorflow as tf

tf.reset_default_graph()

with tf.variable_scope("test1", ):
    var1 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)
    
    with tf.variable_scope("test2"):
        var2 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)

print("var1:", var1.name)
print("var2:", var2.name)

# 通过tf.variable_scope("test1",reuse=True )中的reuse=True设置为True,创建共享变量
with tf.variable_scope("test1", reuse=True):
    var3 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)
    with tf.variable_scope("test2"):
        var4 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)

print("var3:", var3.name)
print("var4:", var4.name)
