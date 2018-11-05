# -*- coding：utf-8 -*-
# &Author  AnFany

import AnFany_DT_Classify as model
import Irisdata_DT_Anfany as dtda

# 最终的函数
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
import matplotlib.pyplot as plt


# 根据不同的深度。看精确率的变化
if __name__ == '__main__':
    # 根据树的不同的初始深度，看正确率的变化
    xunliande = []
    yazhengde = []
    yucede = []

    for shendu in range(2, 13):

        uu = model.DT(train_dtdata=dtda.dt_data, pre_dtdata=dtda.test_data, tree_length=shendu)
        # 完全成长的树
        uu.grow_tree()
        # 剪枝形成的树的集
        gu = uu.prue_tree()
        # 交叉验证形成的最好的树
        cc = uu.jiaocha_tree(gu[0])
        # 根据最好的树预测新的数据集的结果
        uu.noderela = cc[0]
        prenum = uu.pre_tree(uu.pre_dtdata)

        # 验证的
        yazhengde.append(cc[1])
        # 预测的
        yucede.append(uu.compuer_correct(uu.pre_dtdata[:, -1], prenum))
        # 训练
        trainnum = uu.pre_tree(uu.train_dtdata)
        xunliande.append(uu.compuer_correct(uu.train_dtdata[:, -1], trainnum))

        print(xunliande, yazhengde, yucede)

        print('dddddddddddddddddddd', shendu)

    # 绘制图
    plt.plot(list(range(2, 13)), xunliande, 'o--', label='训练', lw=3)
    plt.plot(list(range(2, 13)), yazhengde, '*--', label='验证', lw=3)
    plt.plot(list(range(2, 13)), yucede, 's--', label='预测', lw=3)
    plt.xlabel('树的初始深度')
    plt.xlim(1, 14)
    plt.ylim(min([min(xunliande), min(yazhengde), min(yucede)]) - 0.1, max([max(xunliande), max(yazhengde), max(yucede)]) + 0.1)
    plt.ylabel('正确率')
    plt.grid()
    plt.legend(shadow=True, fancybox=True)
    plt.show()