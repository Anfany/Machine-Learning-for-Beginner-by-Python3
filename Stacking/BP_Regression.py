#-*- coding：utf-8 -*-
# &Author  AnFany

# 适用于多维输出
import numpy as np
import tensorflow as tf

'''基于TensorFlow构建训练函数'''
# 创建激活函数
def activate(input_layer, weights, biases, actfunc):
    layer = tf.add(tf.matmul(input_layer, weights), biases)
    if actfunc == 'relu':
        return tf.nn.relu(layer)
    elif actfunc == 'tanh':
        return tf.nn.tanh(layer)
    elif actfunc == 'sigmoid':
        return tf.nn.sigmoid(layer)
# 权重初始化的方式和利用激活函数的关系很大
# sigmoid: xavir  tanh: xavir    relu: he

#  构建训练函数
def Ten_train(xdata, ydata, prexdata, preydata, hiddenlayers=3, hiddennodes=100, \
              learn_rate=0.05, itertimes=100000, batch_size=200, activate_func='sigmoid', break_error=0.0043):
    # 开始搭建神经网络
    Input_Dimen = len(xdata[0])
    Unit_Layers = [Input_Dimen] + [hiddennodes] * hiddenlayers + [len(ydata[0])]  # 输入的维数，隐层的神经数，输出的维数1

    # 创建占位符
    x_data = tf.placeholder(shape=[None, Input_Dimen], dtype=tf.float32, name='x_data')
    y_target = tf.placeholder(shape=[None, len(ydata[0])], dtype=tf.float32)

    # 实现动态命名变量
    VAR_NAME = locals()

    for jj in range(hiddenlayers + 1):
        VAR_NAME['weight%s' % jj] = tf.Variable(np.random.rand(Unit_Layers[jj], Unit_Layers[jj + 1]), dtype=tf.float32,\
                                                name='weight%s' % jj) / np.sqrt(Unit_Layers[jj])  # sigmoid  tanh
        # VAR_NAME['weight%s'%jj] = tf.Variable(np.random.rand(Unit_Layers[jj], Unit_Layers[jj + 1]), dtype=tf.float32,name='weight%s' % jj) \/ np.sqrt(Unit_Layers[jj] / 2)  # relu
        VAR_NAME['bias%s' % jj] = tf.Variable(tf.random_normal([Unit_Layers[jj + 1]], stddev=10, name='bias%s' % jj),
                                              dtype=tf.float32)
        if jj == 0:
            VAR_NAME['ooutda%s' % jj] = activate(x_data, eval('weight%s' % jj), eval('bias%s' % jj), actfunc=activate_func)
        else:
            VAR_NAME['ooutda%s' % jj] = activate(eval('ooutda%s' % (jj - 1)), eval('weight%s' % jj), \
                                                 eval('bias%s' % jj), actfunc=activate_func)

    # 均方误差
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_target - eval('ooutda%s' % (hiddenlayers))), reduction_indices=[1]))

    # 优化的方法
    my_opt = tf.train.AdamOptimizer(learn_rate)
    train_step = my_opt.minimize(loss)

    # 初始化
    init = tf.global_variables_initializer()

    # 存储误差的字典
    accudict = {}

    loss_vec = []  # 训练误差

    loss_pre = []  # 验证数据误差
    accunum = np.inf
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(init)
        for i in range(itertimes):
            rand_index = np.random.choice(len(xdata), size=batch_size, replace=False)
            rand_x = xdata[rand_index]
            rand_y = ydata[rand_index]

            sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

            temp_loss = sess.run(loss, feed_dict={x_data: xdata, y_target: ydata})

            temmp_losspre = sess.run(loss, feed_dict={x_data: prexdata, y_target: preydata})

            loss_vec.append(temp_loss)
            loss_pre.append(temmp_losspre)

            accudict[i] = [temp_loss, temmp_losspre]

            # 根据输出的误差，判断训练的情况
            if (i + 1) % 20 == 0:
                print('Generation: ' + str(i + 1) + '. 归一训练误差：Loss = ' + str(temp_loss) +
                      '. 归一验证误差：Loss = ' + str(temmp_losspre))

            # 提前退出的判断
            if temp_loss < break_error:  # 根据经验获得此数值, 因为采用的是随机下降，因此误差在前期可能出现浮动
                break

            # 在所有的循环次数中，找到综合误差最低的一次，保存参数
            zongheaccu = 0.01 * temp_loss + 0.99 * temmp_losspre
            if zongheaccu < accunum:
                accunum = zongheaccu
                # 保存模型
                saver.save(sess, './pm25', global_step=i)  # 注意路径

        sign = min(accudict.items(), key=lambda d: 0.01 * d[1][0] + 0.99 * d[1][1])[0]

        return loss_vec, loss_pre, sign, hiddenlayers




