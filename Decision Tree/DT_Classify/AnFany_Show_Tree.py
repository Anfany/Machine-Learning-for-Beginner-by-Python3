# -*- coding：utf-8 -*-
# &Author  AnFany


# 自适应优化绘制决策树程序

# 绘制决策图主要包括四部分
# 1，确定每一个节点展示的内容(内部节点展示，节点名称，类别比例，分类特征，本节点的结果, 叶子节点没有分类特征的内容)
# 2，确定每一个节点的位置（垂直方向平均分配，水平方向按照这一层的节点个数平均分配）
# 3，确定节点之间的连线
# 4，展示连线的内容（分类规则以及分分割值）
# 5，内部节点，子节点以不用的颜色展示，对给出图例


# 根据所有节点的数据集、所有节点的结果、所有节点的规则、剪枝后代表着树的节点关系绘制树
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
import matplotlib.pyplot as plt

# 引入绘制树需要的信息
import AnFany_DT_Classify as tree

# 获得数据的字段名称
ziduan = ['Age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',\
          'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

'''准备部分'''
#  要展示的所有的节点
def allnodes(guanxi):
    allnode = list(guanxi.keys())
    for jj in guanxi:
        for hhh in guanxi[jj]:
            if hhh not in allnode:
                allnode.append(hhh)
    #  之所以要按顺序输出，是因为先画父节点，后画子节点，可以将箭头盖住，更为美观
    return sorted(allnode)

# 要展示的所有的叶子节点
def leafnodes(guanxi):
    allnode = list(guanxi.keys())
    leafnode = []
    for jj in guanxi:
        for hhh in guanxi[jj]:
            if hhh not in allnode:
                leafnode.append(hhh)
    return leafnode

# 要展示的所有的内部节点
def noye_node(guanxi):
    return list(guanxi.keys())


'''第一部分：展示内容'''
# 根据数据集输出各类别之间的比值
def output(shujuji, guanxi):
    #  字典
    leibie = {}
    for jjj in allnodes(guanxi):
        leibie[jjj] = []
        cu = list(shujuji[jjj][:, -1])
        gu = sorted(list(set(list(shujuji[jjj][:, -1]))))
        for du in gu:
            leibie[jjj].append([du, cu.count(du)])  # 各个类别及其数量
    return leibie

# 节点数据集、节点结果、节点规则绘制树
# 制作节点里面的内容
def dingyistr(shujuji, reeult, guize, guanxi, zian=ziduan):
    # 规则字典
    guizezidian = {}
    #  类别字典
    leibii = output(shujuji, guanxi)
    # 字符串字典
    strdict = {}
    # 内部节点
    nonode = noye_node(guanxi)
    # 遍历需要展示的每一个节点，获得每一个节点需展示的字符串内容
    for jjj in allnodes(guanxi):
        # 为节点添加名称
        strdict[jjj] = '节点：%s \n' % jjj  # 内容分行
        # 如果不是内部节点，则不需要添加特征，只添加各个类别的比例
        if jjj not in nonode:
            hu = '占比：'
            for fu in leibii[jjj]:
                hu += '%d:' % fu[1]
            strdict[jjj] += '%s \n' % hu[:-1]
        #  对于内部节点需要多填加一个分类特征的内容、和规则
        else:
            hu = '占比：'
            for fu in leibii[jjj]:
                hu += '%d:' % fu[1]
            strdict[jjj] += '%s \n' % hu[:-1]
            # 添加分类特征
            strdict[jjj] += '特征：%s \n' % zian[guize['%s' % (jjj + 'r')][-1][0]]

            # 添加规则
            sign = 0
            try:
                guize['%s' % (jjj + 'r')][-1][1] + 1
                sign = 1
            except TypeError:
                pass
            if sign == 0:
                guizezidian[jjj + 'l'] = '值为：\n %s' % guize['%s' % (jjj + 'r')][-1][1]
                guizezidian[jjj + 'r'] = '值不为：\n %s' % guize['%s' % (jjj + 'r')][-1][1]
            else:
                guizezidian[jjj + 'l'] = '值不大于：\n %s' % guize['%s' % (jjj + 'r')][-1][1]
                guizezidian[jjj + 'r'] = '值大于：\n %s' % guize['%s' % (jjj + 'r')][-1][1]

        # 为需要展示的节点添加结果
        strdict[jjj] += '结果：%s ' % reeult[jjj]
    return strdict, guizezidian  # 分别返回节点展示的内容字典、连线上需要展示的内容字典


'''第二部分：节点的位置'''
# 根据节点名称的最大长度，确定画布的大小
def huabu(guanxi):
    # 获得所有的节点
    suoyounodes = allnodes(guanxi)
    # 获取最长节点名称字符串的长度，这个长度同时也是树的深度。
    changdu = max(len(i) for i in suoyounodes)
    # 返回长度以及画布大小
    return changdu + 1, 2**max(6, changdu)


# 水平放下的位置，是根据这一层节点的个数、以及此节点的顺序确定的
def getorder(exnode, guanxi):
    fu = []
    for jj in allnodes(guanxi):
        if len(jj) == len(exnode):
            fu.append(jj)
    # 排序
    sfu = sorted(fu)
    return len(sfu) + 1, sfu.index(exnode) + 1 #前者加1是计算间隔，后者加1是因为index从0开始


# 根据画布大小定义每一个节点的横纵坐标位置
def jiedian_location(guanxi):
    # 树的深度，画布大小
    shushen, huahuabu = huabu(guanxi)

    # 返回每个节点坐标的字典
    loca = {}
    # 首先将节点名称按照长度组成字典
    changdu = {}
    for jj in allnodes(guanxi):
        try:
            changdu[len(jj)].append(jj)
        except KeyError:
            changdu[len(jj)] = [jj]
    # 开始确定需要展示节点的位置
    for fi in allnodes(guanxi):
        if fi not in loca:
            for gu in changdu[len(fi)]:  # 同层的节点（也就是节点名称长度一样的）一起计算
                number = getorder(gu, guanxi)
                loca[gu] = [huahuabu / number[0] * number[1], huahuabu - (huahuabu / shushen) * len(gu)]
    return loca

'''第三部分：准备工作结束，开始绘图'''

# 开始绘图
def draw_tree(shujuji, result, guize, guanxi):
    # 字符串内容
    strziu = dingyistr(shujuji, result, guize, guanxi)
    # 节点的位置
    weihzi = jiedian_location(guanxi)

    noyye = noye_node(guanxi)

    # 画布的设置
    huab = huabu(guanxi)[1] + 2  # 上下左右预留空间

    fig, ax = plt.subplots(figsize=(huab, huab))
    # 开始绘制
    for jj in allnodes(guanxi):
        print(jj)
        # 绘制所有的节点要展示的内容
        # 内部节点
        if jj in noyye:
            ax.text(weihzi[jj][0], weihzi[jj][1], strziu[0][jj], size=10, rotation=0.,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round",
                              ec=(1., 0.5, 0.5),
                              fc=(0.2, 0.4, 0.6),
                              )
                    )
        # 叶子节点
        else:
            ax.text(weihzi[jj][0], weihzi[jj][1], strziu[0][jj], size=10, rotation=0.,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round",
                              ec=(1., 0.5, 0.5),
                              fc=(0.6, 0.4, 0.5),
                              )
                    )

        # 只对内部节点绘制箭头和左右的分类规则
        if jj in noyye:
            # 添加左右箭头

            ax.annotate(' ', xy=(weihzi[jj + 'r'][0], weihzi[jj + 'r'][1]), xytext=(weihzi[jj][0], weihzi[jj][1]), ha="center", va="center",
                        arrowprops=dict(facecolor='red', shrink=0.15))

            ax.annotate(' ', xy=(weihzi[jj + 'l'][0], weihzi[jj + 'l'][1]), xytext=(weihzi[jj][0], weihzi[jj][1]),
                        ha="center", va="center", arrowprops=dict(facecolor='red', shrink=0.15))


            # 添加左右规则
            ax.text((weihzi[jj + 'l'][0] + weihzi[jj][0]) / 2, \
                    (weihzi[jj + 'l'][1] + weihzi[jj][1]) / 2, strziu[1][jj + 'l'], fontsize=8)

            ax.text((weihzi[jj + 'r'][0] + weihzi[jj][0]) / 2, \
                    (weihzi[jj + 'r'][1] + weihzi[jj][1]) / 2, strziu[1][jj + 'r'], fontsize=8)

    ax.set(xlim=(0, huab), ylim=(0, huab))

    plt.show()

# 根据不同的深度。看精确率的变化
if __name__ == '__main__':
    # 获得树的信息

    decision_tree = tree.DT()
    # 完全成长的树
    decision_tree.grow_tree()
    # 剪枝形成的树的集
    gu = decision_tree.prue_tree()
    # 交叉验证形成的最好的树
    cc = decision_tree.jiaocha_tree(gu[0])
    print(cc[0])
    # 数据集
    shuju = decision_tree.node_shujuji
    # 结果
    jieguo = decision_tree.jieguo_tree()
    # 规则
    rule = decision_tree.node_rule
    draw_tree(shuju, jieguo, rule, cc[0])
