# -*- coding：utf-8 -*-
# &Author  AnFany


from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import AnFany_DT_Regression as model # 决策树模型
import Data_DT_Regression as datafunc

# 全部数据
x1 = np.arange(10, 200, 0.5)
x2 = np.arange(1, 20, 0.05)
y = np.sin(x1) + np.cos(x2)


XY = np.array([x1, x2, y]).T

# 数据分为训练、验证和预测数据
data_shili = datafunc.fenge(XY)

# 数据
dt_data = data_shili[0]
test_data = data_shili[1]


# 主函数

# 根据不同的深度。看精确率的变化
if __name__ == '__main__':
    # 根据树的不同的初始深度，看正确率的变化
    xunliande = []
    yazhengde = []
    yucede = []

    for shendu in range(2, 25):

        uu = model.DT(train_dtdata=dt_data, pre_dtdata=test_data, tree_length=shendu)
        # 完全成长的树
        uu.grow_tree()

        # 验证的
        yannum = uu.pre_tree(uu.test_dtdata)
        yazhengde.append(uu.compuer_mse(uu.test_dtdata[:, -1], yannum))

        # 预测的
        prenum = uu.pre_tree(uu.pre_dtdata)
        yucede.append(uu.compuer_mse(uu.pre_dtdata[:, -1], prenum))

        # 训练
        trainnum = uu.pre_tree(uu.train_dtdata)
        xunliande.append(uu.compuer_mse(uu.train_dtdata[:, -1], trainnum))

        print(xunliande, yazhengde, yucede)

        print('树的深度', shendu)


    # 在其中选择综合MSE最小的，绘制训练、验证、预测的数据对比
    # 随着树的深度的增加，训练数据的MSE一直在减少
    # 因为没有了剪枝这一步骤，验证和预测数据的意义是一样的
    # 因此当验证和预测的精度不再降低时的深度是最优深度
    zonghe = [j + k for j, k in zip(yazhengde, yucede)]
    # 选择最小的值,
    zuiyoushendu = zonghe.index(min(zonghe)) + 2


    # 绘制不同树的深度的MSE对比图
    plt.plot(list(range(2, 25)), xunliande, 'o--', label='训练', lw=2)
    plt.plot(list(range(2, 25)), yazhengde, '*--', label='验证', lw=2)
    plt.plot(list(range(2, 25)), yucede, 's--', label='预测', lw=2)
    plt.xlabel('树的深度')
    plt.xlim(1, 25)
    plt.title('树的最佳深度为：%d' % zuiyoushendu)
    plt.ylabel('MSE')
    plt.legend(shadow=True, fancybox=True)
    plt.show()


    # 重新建立树
    reuu = model.DT(train_dtdata=dt_data, pre_dtdata=test_data, tree_length=zuiyoushendu)
    # 完全成长的树
    reuu.grow_tree()


    # 验证的
    yannum = reuu.pre_tree(reuu.test_dtdata)


    # 预测的
    prenum = reuu.pre_tree(reuu.pre_dtdata)


    # 训练
    trainnum = reuu.pre_tree(reuu.train_dtdata)



    # 绘制真实值和预测值得曲线
    plt.subplot(211)
    a, b = model.selet(trainnum, reuu.train_dtdata[:, -1], count=100000000)
    plt.plot(list(range(len(a))), a, 'o--', list(range(len(b))), b, '*-')

    plt.legend(['预测', '真实'])
    plt.title('训练数据')


    plt.subplot(223)
    c, d = model.selet(yannum, reuu.test_dtdata[:, -1], count=100000000)
    plt.plot(list(range(len(c))), c, 'o--', list(range(len(d))), d, '*-')
    plt.legend(['预测', '真实'])
    plt.title('验证数据')

    plt.subplot(224)
    e, f = model.selet(prenum, reuu.pre_dtdata[:, -1], count=100000000)
    plt.plot(list(range(len(e))), c, 'o--', list(range(len(d))), f, '*-')
    plt.legend(['预测', '真实'])
    plt.title('预测数据')

    plt.show()




