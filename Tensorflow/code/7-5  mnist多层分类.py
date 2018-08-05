# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # 读取数据

'''一. 定义网络参数'''
# 参数设置
learning_rate = 0.001
training_epochs = 25
batch_size = 100
display_step = 1
# Network Parameters
n_hidden_1 = 256  # 1st layer number of features
n_hidden_2 = 256  # 2nd layer number of features
n_input = 784  # MNIST data 输入 (img shape: 28*28)
n_classes = 10  # MNIST 列别 (0-9 ，一共10类)

'''二. 定义网络结构'''
# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model 构建模型
def multilayer_perceptron(x, weights, biases):
    '''封装网络模型函数,第一层与第二层都使用ReLU激活函数'''
    # 第一层隐藏层
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # 第二层隐藏层
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # 输出层
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# 学习参数(权重 和 偏执)
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# 使用模型multilayer_perceptron 计算输出值
pred = multilayer_perceptron(x, weights, biases)

# 定义损失loss 和 优化器
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

'''三. 运行Session输出结果'''
# 初始化变量
init = tf.global_variables_initializer()
# 启动session
with tf.Session() as sess:
    sess.run(init)

    # 启动循环开始训练 ,迭代training_epochs次
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # 遍历全部数据集
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss 计算平均损失
            avg_cost += c / total_batch
        # 显示训练中的详细信息
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    print(" Finished!")

    # 测试 model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))



