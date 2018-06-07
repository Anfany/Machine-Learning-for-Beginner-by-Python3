# -*- coding：utf-8 -*-
# &Author  AnFany

'''第一部分：库'''
import tensorflow as tf
import BPNN_Classify_Data as bpd
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 中文字体名称
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
# 设置正确率的刻度与子刻度
y_toge = MultipleLocator(0.02)  # 将y轴主刻度标签设置为0.1的倍数
y_son = MultipleLocator(0.01)  # 将此y轴次刻度标签设置为0.01的倍数
#  分类数
countclass = 2


'''第二部分函数'''
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


'''第三部分： 基于TensorFlow构建训练函数'''
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
def Ten_train(xdata, ydata, addxdata, addydata, kcount, hiddenlayers=5, hiddennodes=100, \
              learn_rate=0.009, itertimes=50, batch_size=200, activate_func='sigmoid'):
    # 开始搭建神经网络
    Input_Dimen = len(xdata[0])
    Unit_Layers = [Input_Dimen] + [hiddennodes] * hiddenlayers + [len(ydata[0])]  # 输入的维数，隐层的神经数，输出的维数1

    # 创建占位符
    x_data = tf.placeholder(shape=[None, Input_Dimen], dtype=tf.float32, name='x_data')

    print(x_data)

    y_target = tf.placeholder(shape=[None, len(ydata[0])], dtype=tf.float32)

    # 实现动态命名变量
    VAR_NAME = locals()

    for jj in range(hiddenlayers + 1):
        VAR_NAME['weight%s' % jj] = tf.Variable(np.random.rand(Unit_Layers[jj], Unit_Layers[jj + 1]), dtype=tf.float32,\
                                                name='Weight%s' % jj) / np.sqrt(Unit_Layers[jj])  # sigmoid  tanh
        # VAR_NAME['weight%s'%jj] = tf.Variable(np.random.rand(Unit_Layers[jj], Unit_Layers[jj + 1]), dtype=tf.float32,name='weight%s' % jj) \/ np.sqrt(Unit_Layers[jj] / 2)  # relu
        VAR_NAME['bias%s' % jj] = tf.Variable(tf.random_normal([Unit_Layers[jj + 1]], stddev=10, name='Bias%s' % jj),
                                              dtype=tf.float32)
        if jj == 0:
            VAR_NAME['ooutda%s' % jj] = activate(x_data, eval('weight%s' % jj), eval('bias%s' % jj),
                                                 actfunc=activate_func)
        elif jj == hiddenlayers:
            VAR_NAME['ooutda%s' % jj] = activate(eval('ooutda%s' % (jj - 1)), eval('weight%s' % jj),\
                                                 eval('bias%s' % jj), actfunc='linear')
        else:
            VAR_NAME['ooutda%s' % jj] = activate(eval('ooutda%s' % (jj - 1)), eval('weight%s' % jj),\
                                                 eval('bias%s' % jj), actfunc=activate_func)

    #  需要对输出进行softmax计算
    uuu = tf.nn.softmax(eval('ooutda%s' % (hiddenlayers)))

    # 交叉熵函数  此函数自带sigmoid，因此最后一层采用线性激活函数
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
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 存储精确率的字典
        accudict = {}
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


            print('%s代误差： [训练：%.4f, 验证：%.4f], 正确率： [训练：%.4f, 验证：%.4f]' % (i, temp_loss, temp_loss_add, \
                                                                       acc_rughy_train, acc_rughy))
            accudict[i] = [acc_rughy_train, acc_rughy]
            # # 判断提前退出 ， 验证数据集正确率连续三下降
            # if len(acc_vec_add) >= 4:
            #     # 判断连续三次下降
            #     edlist = acc_vec_add[-4:-1]
            #     delist = acc_vec_add[-3:]
            #     sublist = np.array(edlist) - np.array(delist)
            #     if np.all(sublist > 0):
            #         saver.save(sess, './%smodel' % kcount, global_step=i-3)
            #         print('%s代保存完毕' % kcount)
            #         sign = i - 3
            #         break
        #  在所有的循环次数中，找到综合精确度最高的一次，输出
        sign = max(accudict.items(), key=lambda d: 0.1 * d[1][0] + 0.9 * d[1][1])[0]
        saver.save(sess, './%smodel' % kcount, global_step=sign)
        print('%s代保存完毕' % kcount)
    return loss_vec[: sign + 1], loss_vec_add[: sign + 1], acc_vec[: sign + 1], acc_vec_add[: sign + 1], sign, hiddenlayers


'''第四部分：数据'''
DDatadict = bpd.kfold_train_datadict


#  将数据分为输入数据以及输出数据
def divided(data, cgu=countclass):
    indata = data[:, :-cgu]
    outdata = data[:, -cgu:]
    return indata, outdata


#  将数据字典的值转化为训练输入，训练输出，验证输入、验证输出
def transall(listdata, count=countclass):
    trin, trout = divided(listdata[0], count)
    yanin, yanout = divided(listdata[1], count)
    return trin, trout, yanin, yanout


'''第五部分：最终的运行程序'''
if __name__ == "__main__":
    # 存储正确率 训练
    corrsave_train = []
    # 存储正确率 验证
    corrsave_add = []
    # 存储测试集合的正确率
    corrsave_test = []
    TEST_In, TEST_Out = divided(bpd.Test_data.values)
    # 开始K折交叉验证
    for fold in DDatadict:
        TRAIN_In, TRAIN_Out, ADD_In, ADD_Out = transall(DDatadict[fold])
        bpnn = Ten_train(TRAIN_In, TRAIN_Out, ADD_In, ADD_Out, fold)
        #  下载刚才已经保存的模型
        sess = tf.Session()
        saver = tf.train.import_meta_graph("./%smodel-%s.meta" % (fold, bpnn[4]))
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        graph = tf.get_default_graph()
        op_to_restore = graph.get_tensor_by_name("Add_%s:0" % bpnn[5])
        # Add more to the current graph
        w1 = graph.get_tensor_by_name("x_data:0")
        feed_dict = {w1: TEST_In}
        dgsio = sess.run(op_to_restore, feed_dict)

        #  测试数据集正确率
        add_on_op = outvsreal(judge(dgsio), TEST_Out)
        #  清空图
        tf.reset_default_graph()
        #  测试添加
        corrsave_test.append(add_on_op)
        #  训练正确率添加
        corrsave_train.append(bpnn[2][-4])

        #  验证正确率添加
        corrsave_add.append(bpnn[3][-4])

        # 绘制训练数据集与验证数据集的正确率以及误差曲线
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('代数')
        ax1.set_ylabel('误差', color='r')
        plt.plot(list(range(len(bpnn[0]))), bpnn[0], label='训练', color='r', marker='*', linewidth=2)
        plt.plot(list(range(len(bpnn[1]))), bpnn[1], label='验证', color='r', marker='.', linewidth=2)
        ax1.tick_params(axis='y', labelcolor='r')
        legend = ax1.legend(loc='upper center', shadow=True, fontsize='x-large')
        legend.get_frame().set_facecolor('#F0F8FF')
        ax1.grid(True)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        ax2.set_ylabel('正确率', color='b')  # we already handled the x-label with ax1
        plt.plot(list(range(len(bpnn[2]))), bpnn[2], label='训练', color='b', marker='*',
                 linewidth=2)
        plt.plot(list(range(len(bpnn[3]))), bpnn[3], label='验证', color='b', marker='.',
                 linewidth=2)
        ax2.tick_params(axis='y', labelcolor='b')
        legen = ax2.legend(loc='lower center', shadow=True, fontsize='x-large')
        legen.get_frame().set_facecolor('#FFFAFA')
        ax2.grid(True)
        ax2.yaxis.set_major_locator(y_toge)
        ax2.yaxis.set_minor_locator(y_son)


        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title('%s折训练VS验证 结果对比' % fold, fontsize=16)
        plt.savefig(r'C:\Users\GWT9\Desktop\%s_fol8.jpg' % fold)

    # 绘制K次的结果展示
    plt.figure()
    plt.plot(list(range(len(corrsave_train))), corrsave_train, label='训练', color='b', marker='s', linewidth=2)
    plt.plot(list(range(len(corrsave_add))), corrsave_add, label='验证', color='r', marker='8', linewidth=2)
    plt.plot(list(range(len(corrsave_test))), corrsave_test, label='测试', color='k', marker='d', linewidth=2)
    plt.xlabel('折数')
    plt.ylabel('正确率')
    plt.title('绘制K次的不同数据集的结果展示', fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.savefig(r'C:\Users\GWT9\Desktop\last_fol8.jpg')
    plt.show()


