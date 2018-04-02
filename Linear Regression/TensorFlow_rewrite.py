#-*- coding：utf-8 -*-
# &Author  AnFany

# 引入库
import tensorflow as tf

# 参数
#对于没有归一化的数据，一般要设置较小的学习率
def train_tf(xxdata, yydata, learn_rate=0.00002, iter_times=6000, error=1e-9):
    #占位符
    # 预先输入的数据
    x_data = tf.placeholder(shape=[None, len(xxdata[0])], dtype=tf.float32)
    y_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    # 线性回归参数
    Weight = tf.Variable(tf.random_normal(shape=[len(xxdata[0]), 1]))
    Bias = tf.Variable(tf.random_normal(shape=[1, 1]))
    y_out = tf.add(tf.matmul(x_data, Weight), Bias)
    # 损失函数
    cost = tf.reduce_mean(tf.square(y_out - y_data)) #+ reg_term

    # 初始化
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

    #误差存储
    costfunc = []

    for i in range(iter_times):

        sess.run(optimizer, feed_dict={x_data: xxdata, y_data: yydata})

        y_step_out = sess.run(y_out, feed_dict={x_data: xxdata})

        loss = sess.run(cost, feed_dict={y_out: y_step_out, y_data: yydata})

        costfunc.append(loss)

        # 提前结束循环的机制
        if len(costfunc) > 1:
            if 0 < costfunc[-2] - costfunc[-1] < error:
                break
    predata = sess.run(y_out, feed_dict={x_data: xxdata})
    return predata, Weight.eval(session=sess), Bias.eval(session=sess)






