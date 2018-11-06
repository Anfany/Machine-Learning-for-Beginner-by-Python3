# -*- coding：utf-8 -*-
# &Author  AnFany

import AnFany_DT_Classify as model  # 引入模型
import Irisdata_DT_Anfany as dtda   # 引入数据
import AnFany_Show_Tree as tree     # 引入绘制树

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

    # 在所有的初始深度中，选择三者的综合正确率较高的模型，作为最终的优化树。相同的初始深度越小越好

    zonghe = [x + y + yu for x, y, yu in zip(xunliande, yazhengde, yucede)]


    zuiyoushendu = zonghe.index(max(zonghe)) + 2

    ww = model.DT(train_dtdata=dtda.dt_data, pre_dtdata=dtda.test_data, tree_length=zuiyoushendu)
    # 完全成长的树
    ww.grow_tree()
    # 剪枝形成的树的集
    gu = ww.prue_tree()
    # 交叉验证形成的最好的树
    cc = ww.jiaocha_tree(gu[0])

    # 开始绘制
    # 数据集
    shuju = ww.node_shujuji
    # 结果
    jieguo = ww.jieguo_tree()
    # 规则
    rule = ww.node_rule
    # 绘图
    tree.draw_tree(shuju, jieguo, rule, cc[0], zian=['Sepal_length', 'Sepal_width', 'Petal_length', 'Petal_width'
])


