# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

# 网络结构：2维输入 --> 2维隐藏层 --> 1维输出
'''定义变量'''
learning_rate = 1e-4  # 学习率为0.0001
n_input = 2  # 输入是'2'代表两个数
n_label = 1  # 输出是'1'代表最终的结果
n_hidden = 2  # 隐藏层,隐藏层有2个节点
# 输入占位符x,输出为y
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_label])
'''定义学习参数'''
# h1隐藏层,h2输出层
weights = {'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden], stddev=0.1)),
    'h2': tf.Variable(tf.random_normal([n_hidden, n_label], stddev=0.1))}
biases = {'h1': tf.Variable(tf.zeros([n_hidden])), 'h2': tf.Variable(tf.zeros([n_label]))}

'''定义网络模型:
该例中模型的正向结构入口为x,经过与第一层的w相乘再加上b,通过ReLU函数进行激活转化
最终生成layer_1,再将layer_1代入第二层,使用Tanh激活函数生成--最终的输出y_pred.
'''
layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['h1']))
# y_pred = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['h2']),biases['h2']))
'''调整最后一层的激活函数为Relu或是Sigmoid:
(Sigmoid可以，但是Relu陷入了局部最优解，如果迭代次数增到20000，全 0，即梯度丢失。
于是可以使用 Leaky relus，发现在10000、20000、30000时都会进入局部最优解，
但再也不会出现梯度 消失，将迭代次数变为40000时，得到了正确的模型)
'''
# y_pred = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']),biases['h2']))  # 局部最优解
# y_pred = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['h2']),biases['h2']))

# Leaky relus  40000次 ok
layer2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['h2'])
y_pred = tf.maximum(layer2, 0.01 * layer2)

loss = tf.reduce_mean((y_pred - y) ** 2)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

'''构建模拟数据'''
# 生成数据(手动生成异或数据,相同为0,不同为1)
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0], [1], [1], [0]]
X = np.array(X).astype('float32')
Y = np.array(Y).astype('int16')

'''运行Session,生成结果'''
# 加载
sess = tf.InteractiveSession()  # 创建一个新的交互式TensorFlow会话
sess.run(tf.global_variables_initializer())
# 训练,通过迭代10000次,将模型训练出来
for i in range(10000):
    sess.run(train_step, feed_dict={x: X, y: Y})

# 计算预测值,将做好的X数据集放进去生成结果,# 输出：已训练10000次
print(sess.run(y_pred, feed_dict={x: X}))  # 输出结果四舍五入后,与我们定义的输出Y完全吻合

# 查看隐藏层的输出(第一层的结果)
'''
    输出为4行2列的数组,为隐藏层的输出.同样进行四舍五入
第 一 列为隐藏层第 一 个节点的输出
第 二 列为隐藏层第 二 个节点的输出
    最后一层其实是对隐藏层做了 AND 运算
'''
print(sess.run(layer_1, feed_dict={x: X}))
