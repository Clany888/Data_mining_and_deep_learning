# -*- coding: utf-8 -*-
import tensorflow as tf

# --------------------------------交叉熵试验-----------------------------------------------------------------
labels = [[0, 0, 1], [0, 1, 0]]  # 标签(输入的标签是标准的one_hot)
logits = [[2, 0.5, 6],  # 网络输出值
          [0.1, 0, 3]]
'''
用上面那两个值,进行3次试验
二次softmax试验
观察交叉熵
'''
logits_scaled = tf.nn.softmax(logits)  # 第一次softmax函数,将原本logits求和大于1的  转变为求和等于1
logits_scaled2 = tf.nn.softmax(logits_scaled)  # 第二次softmax函数

result1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)  # 第一次softmax函数交叉熵
result2 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits_scaled)  # 第二次softmax函数交叉熵

# 对softmax得到的值,用自建组合的交叉熵公式 -tf.reduce_sum() 去计算交叉熵,也可以得到正确的值
result3 = -tf.reduce_sum(labels * tf.log(logits_scaled), 1)

with tf.Session() as sess:
    print("scaled=", sess.run(logits_scaled))  # 真实转化的softmax值
    print("scaled2=", sess.run(logits_scaled2))  # 经过第二次的softmax后，分布概率会有变化
    
    '''
    (rel1= [ 0.02215516  3.09967351] 由于样本中第一个是跟标签分类相符的,第二个与标签
    分类不符,所以第一个的交叉熵比较小,第二个的交叉熵比较大)
    '''
    print("rel1=", sess.run(result1), "\n")  # 正确的方式
    
    # 如果将softmax变换完的值放进去会，就相当于算第二次softmax的loss，所以会出错
    print("rel2=", sess.run(result2), "\n")
    print("rel3=", sess.run(result3))

# -----------------------------------one_hot试验--------------------------------------------------------------
# 输入的标签也可以不是标准的one_hot
# 标签总概率为1   和原始的labels = [[0, 0, 1], [0, 1, 0]]等价
labels = [[0.4, 0.1, 0.5], [0.3, 0.6, 0.1]]
# 交叉熵   将上面的标签带入交叉熵公式
result4 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
with tf.Session() as sess:
    print("rel4=", sess.run(result4), "\n")
    '''
    rel4= [ 2.17215538  2.76967359]
    对于正确分类的交叉熵和错误分类的交叉熵,二者的结果没有标准的one_hot那么明显
    (可以用 rel4和其他的rel的值对比得到)
    '''

# --------------------sparse交叉熵的使用---------------------------------------------------------------
# sparse交叉熵,  它使用的是非 one_hot的标签
labels = [2, 1]  # 其实是0 1 2 三个类等价第一行 001 第二行 010  对应的索引值
result5 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
with tf.Session() as sess:
    print("rel5=", sess.run(result5), "\n")  # 得到的rel5 与前面的 rel1结果完全一样

# ---------------------------------计算 loss 值 -----------------------------------------------------------
# 注意！！！这个函数的返回值并不是一个数，而是一个向量，
# 如果要求交叉熵loss，我们要对向量求均值，就是对向量再做一步tf.reduce_sum操作
loss = tf.reduce_sum(result1)
with tf.Session() as sess:
    print("loss=", sess.run(loss))

"""
对于rel3 这种已经求得 softmax 的情况求loss，可以把公式进一步简化成：
loss2 = -tf.reduce_sum(labels * tf.log(logits_scaled))
"""
labels = [[0, 0, 1], [0, 1, 0]]
loss2 = -tf.reduce_sum(labels * tf.log(logits_scaled))
with tf.Session() as sess:
    print("loss2=", sess.run(loss2))
