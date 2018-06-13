# -*- coding：utf-8 -*-
# &Author  AnFany

"""第一部分：库"""

import tensorflow as tf
import TensorFlow_BPNN_Classify as TBC
import Mnist_Data as DATA
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 中文字体名称
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
from matplotlib.ticker import MultipleLocator

# 设置正确率的刻度与子刻度
y_toge = MultipleLocator(0.1)  # 将y轴主刻度标签设置为0.5的倍数
y_son = MultipleLocator(0.01)  # 将此y轴次刻度标签设置为0.1的倍数


"""第二部分：数据"""

DDatadict = DATA.kfold(DATA.TRAIN)

"""第三部分：训练"""
if __name__ == "__main__":
    # 存储正确率 训练
    corrsave_train = []
    # 存储正确率 验证
    corrsave_add = []
    # 存储测试集合的正确率
    corrsave_test = []
    TEST_In, TEST_Out = DATA.TestIN, DATA.TestOUT
    # 开始K折交叉验证
    for fold in DDatadict:
        TRAIN_In, TRAIN_Out, ADD_In, ADD_Out = TBC.transall(DDatadict[fold], count=10)
        while 1:
            bpnn = TBC.Ten_train(TRAIN_In, TRAIN_Out, ADD_In, ADD_Out, fold, itertimes=200, hiddenlayers=5,\
                                 learn_rate=0.003, activate_func='tanh')
            if bpnn:
                break
         #  下载刚才已经保存的模型
        graph = tf.train.import_meta_graph("./gu/%smodel-%s.meta" % (fold, bpnn[4]))
        ses = tf.Session()
        graph.restore(ses, tf.train.latest_checkpoint('./gu/'))
        op_to_restore = tf.get_default_graph().get_tensor_by_name("Add_%s:0" % bpnn[5])
        w1 = tf.get_default_graph().get_tensor_by_name("x_data:0")
        feed_dict = {w1: TEST_In}
        dgsio = ses.run(op_to_restore, feed_dict)
        #  测试数据集正确率
        add_on_op = TBC.outvsreal(TBC.judge(dgsio), TEST_Out)
        print('第%s折测试正确率为' % fold, add_on_op)
        #  清空图
        ses.close()
        tf.reset_default_graph()
        #  测试添加
        corrsave_test.append(add_on_op)
        #  训练正确率添加
        corrsave_train.append(bpnn[2][-1])

        #  验证正确率添加
        corrsave_add.append(bpnn[3][-1])

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
        plt.plot(list(range(len(bpnn[2]))), bpnn[2], label='训练', color='b', marker='*', linewidth=2)
        plt.plot(list(range(len(bpnn[3]))), bpnn[3], label='验证', color='b', marker='.', linewidth=2)
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
