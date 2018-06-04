# -*- coding：utf-8 -*-
# &Author  AnFany

"""第一部分：库"""

import AnFany_BPNN_Classify as AF
import Mnist_Data as DATA
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 中文字体名称
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号

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
        TRAIN_In, TRAIN_Out, ADD_In, ADD_Out = AF.transall(DDatadict[fold], count=10)
        bpnn = AF.BPNN(TRAIN_In, TRAIN_Out, ADD_In, ADD_Out)
        bpnn_train = bpnn.train_adam()
        # 验证正确率
        test_outdata = bpnn.predict(TEST_In)
        teeee = AF.outvsreal(AF.judge(test_outdata), TEST_Out)
        print('第%s次验证：最终的测试数据集的正确率为%.4f' % (fold, teeee))
        # 存储K折训练、验证数据集的正确率
        corrsave_train.append(bpnn_train[3][-4])
        corrsave_add.append(bpnn_train[4][-4])
        corrsave_test.append(teeee)
        # 绘制训练数据集与验证数据集的正确率以及误差曲线
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('代数')
        ax1.set_ylabel('误差', color='r')
        plt.plot(list(range(len(bpnn_train[2]))), bpnn_train[2], label='训练', color='r', marker='*', linewidth=2)
        plt.plot(list(range(len(bpnn_train[5]))), bpnn_train[5], label='验证', color='r', marker='.', linewidth=2)
        ax1.tick_params(axis='y', labelcolor='r')
        legend = ax1.legend(loc='upper center', shadow=True, fontsize='x-large')
        legend.get_frame().set_facecolor('#F0F8FF')
        ax1.grid(True)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        ax2.set_ylabel('正确率', color='b')  # we already handled the x-label with ax1
        plt.plot(list(range(len(bpnn_train[3][:-3]))), bpnn_train[3][:-3], label='训练', color='b', marker='*', linewidth=2)
        plt.plot(list(range(len(bpnn_train[4][:-3]))), bpnn_train[4][:-3], label='验证', color='b', marker='.', linewidth=2)
        ax2.tick_params(axis='y', labelcolor='b')
        legen = ax2.legend(loc='lower center', shadow=True, fontsize='x-large')
        legen.get_frame().set_facecolor('#FFFAFA')
        ax2.grid(True)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title('%s折训练VS验证 结果对比' % fold, fontsize=16)
        plt.savefig(r'C:\Users\GWT9\Desktop\%s_foldui.jpg' % fold)

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
    plt.savefig(r'C:\Users\GWT9\Desktop\last_foldui.jpg')
    plt.show()
