# -*- coding: utf-8 -*-
import tensorflow as tf

global_step = tf.Variable(0, trainable=False)

initial_learning_rate = 0.1  # 初始学习率

# 采用退化学习率
learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps=10, decay_rate=0.9)
opt = tf.train.GradientDescentOptimizer(learning_rate)  # 创建优化器

add_global = global_step.assign_add(1)  # 定义op,令global_step每次+1 完成计步     assign_add 自增1
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(learning_rate))  # 输出刚开始的学习率 0.1
    for i in range(20):
        '''循环20次,将每次的学习率打印出来'''
        g, rate = sess.run([add_global, learning_rate])
        print(g, rate)
