# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

# 1 创建图的方法
c = tf.constant(0.0)

g = tf.Graph()
with g.as_default():  # as_default将此图作为默认图
    c1 = tf.constant(0.0)
    print(c1.name)
    print(c1.graph)  # 图g
    print(g)  # 图g
    print(c.graph)  # 自带的图

g2 = tf.get_default_graph()
print(g2)  # 默认的图

tf.reset_default_graph()  # 重新创建了一张图来代替原来的图
g3 = tf.get_default_graph()
print(g3)
print('-------------------------------------')

# 2.获取tensor

print(c1.name)
t = g.get_tensor_by_name(name="Const:0")
print(t)
print('-------------------------------------')

# 3 获取op
a = tf.constant([[1.0, 2.0]])
b = tf.constant([[1.0], [3.0]])

tensor1 = tf.matmul(a, b, name='exampleop')
print(tensor1.name, tensor1)
test = g3.get_tensor_by_name("exampleop:0")  # 获取张量
print(test)

print(tensor1.op.name)
testop = g3.get_operation_by_name("exampleop")  # 根据操作name,获取节点操作
print(testop)

with tf.Session() as sess:
    test = sess.run(test)
    print(test)
    test = tf.get_default_graph().get_tensor_by_name("exampleop:0")
    print(test)
print('-------------------------------------')

# 4. 获取所有列表

# 返回图中的操作节点列表
tt2 = g.get_operations()
print(tt2)  # >>>[<tf.Operation 'Const' type=Const>]  因为图中只有一个常量c1 ,所以只打印出一条
# 5. 获取对象
tt3 = g.as_graph_element(c1)
print(tt3)  # Tensor("Const:0", shape=(), dtype=float32)
print("________________________\n")

# 练习
with g.as_default():
    c1 = tf.constant(0.0)
    print(c1.graph)
    print(g)
    print(c.graph)
    g3 = tf.get_default_graph()
    print(g3)
