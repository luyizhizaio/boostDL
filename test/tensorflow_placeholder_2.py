__author__ = 'dayue'

import  tensorflow as tf

#定义变量,随机变量 标准差为1，均值为0的2*3矩阵
w1 = tf.Variable(tf.random_normal([2,3],stddev=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1))

#定义 placeholder 作为存放输入数据的地方，
#x = tf.placeholder(tf.float32,shape=(1,2),name="input")
x = tf.placeholder(tf.float32,shape=(3,2),name="input")
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

sess = tf.Session()
#初始化所有变量
init_op = tf.initialize_all_variables()

sess.run(init_op)

#使用feed_dict来指定x的取值，
#print(sess.run(y,feed_dict={x:[[0.7,0.9]]}))
#[[-1.11611199]]
#提供批量的训练数据



print(sess.run(y,feed_dict={x:[[.7,.9],[.1,.4],[.5,.8]]}))
# [[ 7.18358898]
#  [ 3.11917114]
#  [ 6.34283972]]






