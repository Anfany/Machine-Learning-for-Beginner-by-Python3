# -*- coding：utf-8 -*-
# &Author  AnFany

import tensorflow as tf
sess = tf.Session()

# 构建函数
def trans_tf(datax, datay, prea, learn_rate=0.5, iter_tiems=40000, error=1e-9, con='L2'):
    # 占位符
    x_data = tf.placeholder(shape=[None, len(datax[0])], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    # 逻辑回归参数
    Weight = tf.Variable(tf.random_normal(shape=[len(datax[0]), 1]), dtype=tf.float32)
    Bias = tf.Variable(tf.random_normal(shape=[1, 1]), dtype=tf.float32)

    model_output = tf.add(tf.matmul(x_data, Weight), Bias)

    tf.add_to_collection(tf.GraphKeys.WEIGHTS, Weight)
    regularizer = tf.contrib.layers.l2_regularizer(scale=2 / 2000)
    reg_term = tf.contrib.layers.apply_regularization(regularizer)

    # 正则化
    if con == 'L2':
        # 损失函数
        costfunc = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target)) + reg_term
    else:
        costfunc = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))

    # 利用不同的方法
    #optimizer = tf.train.GradientDescentOptimizer(learn_rate) # 梯度
    #optimizer = tf.train.MomentumOptimizer(learning_rate=learn_rate, momentum=0.9) # 动量法
    #optimizer = tf.train.AdadeltaOptimizer(learning_rate=learn_rate, rho=0.55, epsilon=1e-08)
    optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate, beta1=0.9, beta2=0.99, epsilon=1e-08)
    trainstep = optimizer.minimize(costfunc)
    init = tf.global_variables_initializer()
    sess.run(init)

    loss_vec = []  # 存储成本函数值
    # 开始训练
    for i in range(iter_tiems):
        sess.run(trainstep, feed_dict={x_data: datax, y_target: datay})
        temp_loss = sess.run(costfunc, feed_dict={x_data: datax, y_target: datay})
        loss_vec.append(temp_loss)
        # 训练提前结束
        if len(loss_vec) > 2:
            if loss_vec[-2] - loss_vec[-1] >= 0 and (loss_vec[-2] - loss_vec[-1]) <= error:
                break

    predata = sess.run(model_output, feed_dict={x_data: prea})

    #转化
    trans_predata = [[1] if jj[0] >= 0 else [0] for jj in predata]

    return loss_vec, trans_predata
