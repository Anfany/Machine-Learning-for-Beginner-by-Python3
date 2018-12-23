# -*- coding：utf-8 -*-
# &Author  AnFany

'''第一部分：库'''
import tensorflow as tf
import numpy as np

'''函数'''
#  根据输出的结果判断类别的函数
def judge(ydata):
    maxnum = np.max(ydata, axis=1)
    lastdata = []
    for ii in range(len(ydata)):
        maxindex = list(ydata[ii]).index(maxnum[ii])
        fu = [0] * len(ydata[0])
        fu[maxindex] = 1
        lastdata.append(fu)
    return np.array(lastdata)

#  根据输出的结果以及真实结果输出分类的效果
def outvsreal(outdata, realdata):
    subdata = outdata - realdata
    sundata = np.sum(np.abs(subdata), axis=1)
    correct = list(sundata).count(0)
    return correct / len(outdata)


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
    elif actfunc == 'linear':
        return layer


# 权重初始化的方式和利用激活函数的关系很大
# sigmoid: xavir  tanh: xavir   relu: he

#  构建训练函数
def Ten_train(xdata, ydata, addxdata, addydata, hiddenlayers=3, hiddennodes=100, \
              learn_rate=0.02, itertimes=20, batch_size=200, activate_func='tanh'):
    # 开始搭建神经网络
    Input_Dimen = len(xdata[0])
    Unit_Layers = [Input_Dimen] + [hiddennodes] * hiddenlayers + [len(ydata[0])]  # 输入的维数，隐层的神经数，输出的维数1

    # 创建占位符
    x_data = tf.placeholder(shape=[None, Input_Dimen], dtype=tf.float32, name='x_data')

    y_target = tf.placeholder(shape=[None, len(ydata[0])], dtype=tf.float32)

    # 实现动态命名变量
    VAR_NAME = locals()
    for jj in range(hiddenlayers + 1):
        VAR_NAME['weight%s' % jj] = tf.Variable(np.random.rand(Unit_Layers[jj], Unit_Layers[jj + 1]) / np.sqrt(Unit_Layers[jj]), \
                                                dtype=tf.float32, name='Weight%s' % jj)  # sigmoid  tanh
        # VAR_NAME['weight%s'%jj] = tf.Variable(np.random.rand(Unit_Layers[jj], Unit_Layers[jj + 1]),dtype=tf.float32, \name='weight%s' % jj) \/ np.sqrt(Unit_Layers[jj] / 2)  # relu
        VAR_NAME['bias%s' % jj] = tf.Variable(tf.random_normal([Unit_Layers[jj + 1]], stddev=10), dtype=tf.float32, name='Bias%s' % jj)
        if jj == 0:
            VAR_NAME['ooutda%s' % jj] = activate(x_data, eval('weight%s' % jj), eval('bias%s' % jj),
                                                 actfunc=activate_func)
        elif jj == hiddenlayers:
            VAR_NAME['ooutda%s' % jj] = activate(eval('ooutda%s' % (jj - 1)), eval('weight%s' % jj),\
                                                 eval('bias%s' % jj), actfunc='linear')  # 因此最后一层采用线性激活函数
        else:
            VAR_NAME['ooutda%s' % jj] = activate(eval('ooutda%s' % (jj - 1)), eval('weight%s' % jj),\
                                                 eval('bias%s' % jj), actfunc=activate_func)
    #  需要对输出进行softmax计算
    uuu = tf.nn.softmax(eval('ooutda%s' % (hiddenlayers)))

    # 交叉熵函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_target, logits=eval('ooutda%s' % (hiddenlayers))))

    # 计算精确度需要

    accu = eval('ooutda%s' % hiddenlayers)

    # 优化的方法
    # my_opt = tf.train.GradientDescentOptimizer(learn_rate)
    my_opt = tf.train.AdamOptimizer(learn_rate)
    train_step = my_opt.minimize(loss)

    # 初始化
    init = tf.global_variables_initializer()

    loss_vec = []  # 训练误差

    loss_vec_add = []  # 验证误差

    acc_vec = []  # 训练精确度

    acc_vec_add = []  # 验证精确度

    #  需要保存的权重以及偏置
    graph = tf.get_default_graph()
    saver = tf.train.Saver(max_to_keep=1)
    sess = tf.Session()
    # 存储精确率的字典
    accudict = {}
    accunum = 0
    sess.run(init)
    for i in range(itertimes):  # 在总共的迭代次数中选择最高的（验证正确率+训练精确率）
        for jj in range(int(len(xdata) / batch_size)):
            rand_index = np.random.choice(len(xdata), size=batch_size, replace=False)
            rand_x = xdata[rand_index]
            rand_y = ydata[rand_index]
            #  开始训练
            sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

        #  训练误差
        temp_loss = sess.run(loss, feed_dict={x_data: xdata, y_target: ydata})
        #  存储训练误差
        loss_vec.append(temp_loss)

        #  验证误差
        temp_loss_add = sess.run(loss, feed_dict={x_data: addxdata, y_target: addydata})
        #  存储验证误差
        loss_vec_add.append(temp_loss_add)

        # 训练精确率
        acc_ru = sess.run(accu, feed_dict={x_data: xdata})
        acc_rughy_train = outvsreal(judge(acc_ru), ydata)
        #  存储
        acc_vec.append(acc_rughy_train)
        #  验证精确率
        acu = sess.run(accu, feed_dict={x_data: addxdata})
        acc_rughy = outvsreal(judge(acu), addydata)
        # 存储
        acc_vec_add.append(acc_rughy)

        print('%s代误差： [训练：%.4f, 验证：%.4f], 正确率： [训练：%.4f, 验证：%.4f]' % (i, temp_loss,
                                                                      temp_loss_add, acc_rughy_train, acc_rughy))
        accudict[i] = [acc_rughy_train, acc_rughy]

        #  在所有的循环次数中，找到综合精确度最高的一次，保存参数
        zongheaccu = 0.1 * acc_rughy_train + 0.9 * acc_rughy
        if zongheaccu > accunum:
            accunum = zongheaccu
            # 保存模型
            saver.save(sess, r'E:\tensorflow_Learn\Stacking\adult\model', global_step=i)  # 注意路径

    sign = max(accudict.items(), key=lambda d: 0.1 * d[1][0] + 0.9 * d[1][1])[0]
    print('折运行完毕，模型已经保存，最优的是%s代' % sign)
    return loss_vec[: sign + 1], loss_vec_add[: sign + 1], acc_vec[: sign + 1], acc_vec_add[: sign + 1], sign, hiddenlayers


