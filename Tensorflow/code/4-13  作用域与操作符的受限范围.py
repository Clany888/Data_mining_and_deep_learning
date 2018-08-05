# -*- coding: utf-8 -*-
import tensorflow as tf

tf.reset_default_graph()  # 清空图里面的全部变量
# 使用 with variable_ scope（" name"） as xxxscope 的 方式 定义 作用域
# 当使用 这种 方式 时， 所 定义 的 作用域 变量 xxxscope 将不 再 受到 外围 的 scope 所 限制。
with tf.variable_scope("scope1") as sp:  # 创建局部作用域,命名为sp
    var1 = tf.get_variable("v", [1])

print("sp:", sp.name)
print("var1:", var1.name)

with tf.variable_scope("scope2"):
    var2 = tf.get_variable("v", [1])
    
    with tf.variable_scope(sp) as sp1:
        var3 = tf.get_variable("v3", [1])
        
        with tf.variable_scope(""):  # var4: scope1//v4:0   因为没有指定,所以默认属于sp作用域中
            var4 = tf.get_variable("v4", [1])
# sp和var1的输出前面已经交代过。
# sp1在scope2下，但是输出仍是scope1，没有改变。
# 在它下面定义的var3的名字是scope1/v3：0，表明也在scope1下，再次说明sp没有受到外层的限制。
print("sp1:", sp1.name)
print("var2:", var2.name)
print("var3:", var3.name)
print("var4:", var4.name)
with tf.variable_scope("scope"):
    with tf.name_scope("bar"):
        v = tf.get_variable("v", [1])
        x = 1.0 + v
        # 注意点
        with tf.name_scope(""):
            y = 1.0 + v
print("v:", v.name)
print("x.op:", x.op.name)
print("y.op:", y.op.name)
