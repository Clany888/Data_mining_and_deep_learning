# -*- coding: utf-8 -*-
import pylab
from tensorflow.examples.tutorials.mnist import input_data

'''
[偏移]   [数据类型]       [值]               [描述]
0000     32位整数        0x00000801(2049)   魔数ID (MSB优先，大端模式)
0004     32位整数        60000              后面共有多少项标签
0008     无符号字节      ??                  标签
0009     无符号字节      ??                  标签
........
xxxx     无符号字节      ??                  标签
标签的值是 0 到 9.
'''

'''
train-images-idx3-ubyte: 训练集合  图片数据
train-labels-idx1-ubyte: 训练集合  标签数据
t10k-images-idx3-ubyte:  测试集合  图片数据
t10k-labels-idx1-ubyte:  测试集合  标签数据
'''
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print('输入数据:', mnist.train.images)
print('输入数据打shape:', mnist.train.images.shape)  # 训练数据集(55000, 784)

im = mnist.train.images[1]
im = im.reshape(-1, 28)
pylab.imshow(im)
pylab.show()

print('输入数据打shape:', mnist.test.images.shape)  # 测试集(10000, 784)
print('输入数据打shape:', mnist.validation.images.shape)  # 验证数据集(5000, 784)
