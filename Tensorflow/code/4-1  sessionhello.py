# -*- coding: utf-8 -*-
import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')  # 定义一个常量
sess = tf.Session()  # 建立一个session
print(sess.run(hello))  # 通过session里面的run来运行结果
sess.close()  # 关闭session
