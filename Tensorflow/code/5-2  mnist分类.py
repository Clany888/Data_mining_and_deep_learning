# -*- coding: utf-8 -*-
import tensorflow as tf
import pylab
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # 读取数据
# one_hot=True  设置为长度为n的数组,只有一个是1.0,其他都是0.0


# --------------------------------正向传播begging--------------------------------------------------
tf.reset_default_graph()  # 初始化所有变量
# tf Graph Input  输入
x = tf.placeholder(tf.float32, [None, 784])  # mnist data维度 28*28=784
y = tf.placeholder(tf.float32, [None, 10])  # 0-9 数字=> 10 classes

# Set model weights  设置权重
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 构建模型
pred = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax分类

# ----------------------------反向传播BEGGING-------------------------------------------------
# 损失函数 Minimize error using cross entropy
# 将生成的pred与样本标签y进行一次交叉熵的运算,然后取平均值
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))

# 定义参数
learning_rate = 0.01  # 学习率
# 使用梯度下降优化器
# 将cost作为一次正向传播的误差, 通过梯度下降的优化方法找到能够使这个误差最小化的b和W的偏移量
# 更新b 和 w,使其调整为合适的参数.  整个过程就是不断让损失值变小,才能表明输出结果跟标签数据越相近.
# 当cost小到我们的需求时,这时的b和w就是训练出来的合适值.
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# ---------------------------训练模型----------------------------------------------------------
training_epochs = 25  # 一共迭代25次
batch_size = 100  # 一次取100 条数据进行训练   (很重要!!!深度学习中都是把数据按批量小部分喂进去进行)
display_step = 1  # 每训练一次就把具体的中间状态显示出来
saver = tf.train.Saver()
model_path = "log/521model.ckpt"

# 1.---------启动Session开始训练----------
# 启动session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # Initializing OP 初始化所有变量
    
    # 启动循环开始训练
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)  # 55000/100
        # 遍历全部数据集
        for i in range(total_batch):
            # mnist.train.next_batch(batch_size)  : Return the next `batch_size` examples from this data set.
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 运行优化器 Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
            # 计算平均loss值  Compute average loss
            avg_cost += c / total_batch
        # 显示训练中的详细信息
        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))  # 输出的中间状态是cost损失值
    print(" Finished!")
    
    # ----------------------------测试模型------------------------------------------------------------
    # 测试 model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    
    # -------------保存模型-----------------
    # Save model weights to disk
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)

# ===================================读取模型========================================================
# 读取模型
print("Starting 2nd session...")
with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    # Restore model weights from previously saved model
    saver.restore(sess, model_path)
    
    # 测试 model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    
    output = tf.argmax(pred, 1)
    batch_xs, batch_ys = mnist.train.next_batch(2)
    outputval, predv = sess.run([output, pred], feed_dict={x: batch_xs})
    print("--输出值--\n", outputval, "\n--预测值--\n", predv, "\n--所属类标签的值--\n", batch_ys)
    
    im = batch_xs[0]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()
    
    im = batch_xs[1]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()
