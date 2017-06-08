__author__ = 'dayue'


import tensorflow as tf
#声明变量
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1, seed=1))

#暂时将输入的特征向量定义为一个常量 ,1*2的矩阵
x = tf.constant([[0.7,0.9]])

#向前传播算法获得神经网络的输出
a = tf.matmul(x,w1) #矩阵相乘
y = tf.matmul(a,w2)

sess = tf.Session()

#初始化两个变量
sess.run(w1.initializer)
sess.run(w2.initializer)

#输出
print(sess.run(y))
sess.close()

