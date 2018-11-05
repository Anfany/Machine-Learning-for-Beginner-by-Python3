# -*- coding：utf-8 -*-
# &Author  AnFany

# CART分类树：可处理连续、离散的变量，支持多分类
# 测试数据和训练数据的字段顺序必须一样，因为本程序在设定规则按的是字段的编号，而不是名字

# 引入数据
import DT_Classify_Data as dtda

import copy
import pandas as pd
import numpy as np

# 定义函数
class DT:
    def __init__(self, train_dtdata=dtda.dt_data, pre_dtdata=dtda.test_data, tree_length=4):

        # 训练数据
        self.train_dtdata = train_dtdata[0]['train']

        # 验证数据
        self.test_dtdata = train_dtdata[0]['test']

        # 预测数据
        self.pre_dtdata = pre_dtdata

        # 中间过程变量
        self.node_shujuji = {'0': self.train_dtdata}  # 存储每一个节点数据集的字典
        self.fenlei_shujuji = {'0': self.train_dtdata}  # 存储需要分类的数据集的字典

        # 叶子节点的集合
        self.leafnodes = []
        # 节点关系字典
        self.noderela = {}
        # 每一个节点的规则关系
        self.node_rule = {'0': []}

        # 避免树过大，采用限制书的深度以及基尼系数的值
        self.tree_length = tree_length


    #  根据类别的数组计算基尼指数
    def jini_zhishu(self, exlist):
        dnum = 0
        leng = len(exlist)
        for hh in list(set(exlist)):
            dnum += (list(exlist).count(hh) / leng) ** 2
        return 1 - dnum

    #  计算基尼系数的函数
    def jini_xishu(self, tezheng, leibie):  # 输入特征数据，类别数据，返回最小基尼系数对应的值
        #  首先判断特征数据是连续、或者是分类的
        sign = 0
        try:
            tezheng[0] + 2
            # 证明是连续的
            sign = 1
        except TypeError:
            pass
        if sign:  # 连续变量
            # 去重、排序
            quzhong = np.array(sorted(list(set(tezheng))))
            # 判断是不是就一个值
            if len(quzhong) == 1:
                return False
            # 取中间值
            midd = (quzhong[:-1] + quzhong[1:]) / 2
            # 开始遍历每一个中间值，计算对应的基尼系数
            length = len(leibie)
            # 存储基尼系数的值
            save_ji, jini = np.inf, 0
            number = ''
            for mi in midd:
                #  计算基尼系数
                onelist = leibie[tezheng <= mi]
                twolist = leibie[tezheng > mi]
                jini = (len(onelist) / length) * self.jini_zhishu(onelist) + (len(twolist) / length) * self.jini_zhishu(twolist)

                if jini <= save_ji:
                    save_ji = jini
                    number = mi
            return number, save_ji
        else:  #分类变量
            # 去重、排序
            quzhong = np.array(list(set(tezheng)))
            # 判断是不是就一个值
            if len(quzhong) == 1:
                return False
            # 开始遍历每一个值，计算对应的基尼系数
            length = len(leibie)
            # 存储基尼系数的值
            jini, save_ji = 0, np.inf
            number = ''
            for mi in quzhong:
                #  计算基尼系数
                onelist = leibie[tezheng == mi]
                twolist = leibie[tezheng != mi]
                jini = (len(onelist) / length) * self.jini_zhishu(onelist) + (len(twolist) / length) * self.jini_zhishu(
                    twolist)
                if jini <= save_ji:
                    save_ji = jini
                    number = mi
            return number, save_ji  # 该特征最好的分割值，以及该特征最小的基尼系数


    # 数据集确定分类特征以及属性的函数
    def feature_zhi(self, datadist):  # 输入的数据集字典，输出最优的特征编号，以及对应的值，还有基尼系数
        tezhengsign = ''
        number = np.inf
        jini = ''
        for jil in range(1, len(datadist[0])):
            #  获取特征数据和类别数据
            tezhen = datadist[:, (jil - 1): jil].T[0]
            leib = datadist[:, -1:].T[0]
            # 在其中选择最小的
            cresu = self.jini_xishu(tezhen, leib)
            # 判断这个特征可不可取
            if cresu:
                if cresu[1] <= number:
                    number = cresu[1]
                    tezhengsign = jil - 1
                    jini = cresu[0]
        if jini != '':
            return tezhengsign, jini, number  # 特征编号, 该特征最好的分割值，该数据集最小的基尼系数
        else:
            return False  # 这个数据集无法被分裂

    # 将数据集合分裂
    def devided_shujuji(self, datadis):  # 输入特征编号，对应的值，返回两个数据集
        # 运算的结果
        yuansuan = self.feature_zhi(datadis)
        if yuansuan:
            #  需要判断这个被选中的特征是连续还是离散的
            try:
                datadis[:, yuansuan[0]][0] + 2
                oneshujui = datadis[datadis[:, yuansuan[0]] <= yuansuan[1]]
                twoshujui = datadis[datadis[:, yuansuan[0]] > yuansuan[1]]
            except TypeError:
                oneshujui = datadis[datadis[:, yuansuan[0]] == yuansuan[1]]
                twoshujui = datadis[datadis[:, yuansuan[0]] != yuansuan[1]]
            return oneshujui, twoshujui, yuansuan
        else:
            return False

    # 决策树函数
    def grow_tree(self):
        while len(self.fenlei_shujuji) != 0:
            # 需要复制字典
            copy_dict = copy.deepcopy(self.fenlei_shujuji)
            # 开始遍历每一个需要分类的数据集
            for hd in self.fenlei_shujuji:
                #  在这里限制树的深度
                if len(hd) == self.tree_length + 1:
                    # 不需要在分裂
                    del copy_dict[hd]
                    # 添加到叶子节点的集合中
                    self.leafnodes.append(hd)
                else:
                    fenguo = self.devided_shujuji(copy_dict[hd])
                    if fenguo:
                        if len(set(fenguo[0][:, -1])) == 1:  # 数据集是一个类别就不再分裂
                            self.leafnodes.append('%sl' % hd)   # 成叶子节点
                        else:
                            copy_dict['%sl' % hd] = fenguo[0]  # 继续分裂

                        self.node_shujuji['%sl' % hd] = fenguo[0]  # 总的数据集

                        # 添加节点的规则
                        self.node_rule['%sl' % hd] = (self.node_rule[hd]).copy()
                        self.node_rule['%sl' % hd].append(fenguo[2])


                        if len(set(fenguo[1][:, -1])) == 1:
                            self.leafnodes.append('%sr' % hd)
                        else:
                            copy_dict['%sr' % hd] = fenguo[1]

                        self.node_shujuji['%sr' % hd] = fenguo[1]

                        # 添加节点的规则
                        self.node_rule['%sr' % hd] = (self.node_rule[hd]).copy()
                        self.node_rule['%sr' % hd].append(fenguo[2])

                        # 添加到节点关系字典
                        self.noderela[hd] = ['%sl' % hd, '%sr' % hd]

                    del copy_dict[hd]  # 需要在分裂数据中删除这一个

            self.fenlei_shujuji = copy.deepcopy(copy_dict)

            print(len(self.fenlei_shujuji))
            print(len(self.node_shujuji))
        print(self.node_rule)
        print(self.noderela)
        return 'done'

    # 根据树得出每一个节点数据集的结果
    def jieguo_tree(self):
        # 根据每一个数据得到每一个节点对应的结果
        shujuji_jieguo = {}
        for shuju in self.node_shujuji:
            zuihang = self.node_shujuji[shuju][:, -1]
            # 选择最多的
            duodict = {ik: list(zuihang).count(ik) for ik in set(list(zuihang))}
            # 在其中选择最多的
            shujuji_jieguo[shuju] = max(duodict.items(), key=lambda dw: dw[1])[0]

        return shujuji_jieguo

    # 要得到叶子节点的集合
    def leafnodes_tree(self):
        # 不在键值中的所有节点
        keynodes = list(self.noderela.keys())
        zhin= list(self.noderela.values())
        zhinodes = []
        for hhu in zhin:
            for fff in hhu:
                zhinodes.append(fff)
        leafnodes = [jj for jj in zhinodes if jj not in keynodes]
        return leafnodes


    # 寻找任何一个内部节点的叶子节点
    def iner_leaf(self, exnode):
        # 内部节点
        inernodes = list(self.noderela.keys())
        # 叶子节点
        llnodes = []
        # 全部的节点
        ghunodes = list(self.noderela.values())

        gugu = []

        for hhdd in ghunodes:
            for ghgh in hhdd:
                gugu.append(ghgh)

        for jj in gugu + ['0']:
            if jj not in inernodes:
                if len(jj) > len(exnode) and exnode in jj:
                    llnodes.append(jj)
        return llnodes

    # 寻找任何一个内部节点的下属的节点
    def xiashu_leaf(self, exnode):
        # 叶子节点
        xiashunodes = []
        # 全部的节点
        godes = list(self.noderela.values())

        gug = []

        for hhdd in godes:
            for ghgh in hhdd:
                gug.append(ghgh)

        for jj in gug + ['0']:
            if exnode in jj:
                xiashunodes.append(jj)
        return xiashunodes


    # 判读数据是否符合这个规矩的函数
    def judge_data(self, data, signstr, guize):
        # 首先判断数据连续或者是离散
        fign = 0
        try:
            data[guize[0]] + 2
            fign = 1
        except TypeError:
            pass
        if fign == 1:  # 连续
            if signstr == 'r':
                if data[guize[0]] > guize[1]:
                    return True
                return False
            elif signstr == 'l':
                if data[guize[0]] <= guize[1]:
                    return True
                return False
        elif fign == 0:  # 离散
            if signstr == 'r':
                if data[guize[0]] != guize[1]:
                    return True
                return False
            elif signstr == 'l':
                if data[guize[0]] == guize[1]:
                    return True
                return False



    # 预测函数, 根据节点的关系字典以及规则、每个节点的结果获得预测数据的结果
    def pre_tree(self, predata):
        # 每个数据集合的结果
        meire = self.jieguo_tree()
        # 存储结果
        savresu = []
        # 首先根据节点关系找到所有的叶子节点
        yezinodes = self.leafnodes_tree()
        # 开始判断数据
        for jj in predata:
            shuju = jj[: -1]
            # 开始判断
            for yy in yezinodes:
                gu = 1
                guide = self.node_rule[yy]
                for iu, ju in zip(yy[1:], guide):
                    if not self.judge_data(shuju, iu, ju):
                        gu = 0
                        break
                if gu == 1:
                    savresu.append(meire[yy])
        return savresu


    # 计算每一个节点的剪枝的基尼系数
    def jianzhi_iner(self, exnode):
        # 首先得到整体训练数据集的长度
        leng = len(self.train_dtdata)
        # # 在得到本节点数据集的长度,此项可以被消去
        # benleng = len(self.node_shujuji[exnode])

        # 计算被错误分类的数据的条数
        self.node_result = self.jieguo_tree()
        cuowu_leng = len(self.node_shujuji[exnode][self.node_shujuji[exnode][:, -1] != self.node_result[exnode]])
        # 计算
        jinum = cuowu_leng / leng
        return jinum

    # 计算每一个内部节点的下属叶子节点的基尼系数之和
    def iner_sum(self, ecnode):
        jnum = 0
        # 首先得到这个内部节点下属的所有叶子节点
        for hhh in self.iner_leaf(ecnode):
            jnum += self.jianzhi_iner(hhh)
        return jnum


    # 树的剪枝， 每一棵树都是一个字典形式（节点关系就代表一棵子树）
    def prue_tree(self):
        # 开始剪枝
        tree_set = {}
        # a值的字典
        adict = {}

        # 第一棵完全生长的树
        sign = 0
        tree_set[sign] = self.noderela.copy()
        # 开始剪枝
        while len(list(self.noderela.keys())) != 0:
            # 复制字典
            coppdict = self.noderela.copy()
            # 存储内部节点剪枝基尼系数的字典
            saveiner = {}
            for jiner in list(self.noderela.keys()):
                # 每一个内部节点计算
                saveiner[jiner] = (self.jianzhi_iner(jiner) - self.iner_sum(jiner)) / (len(self.iner_leaf(jiner)) - 1)
            # 选择其中最小的，如果有2个相同的选择最长的
            numm = np.inf
            dd = ''
            for hji in saveiner:
                if numm > saveiner[hji]:
                    dd = hji
                    numm = saveiner[hji]
                elif numm == saveiner[hji]:
                    if len(dd) < len(hji):
                        dd = hji
            # 添加到a值
            adict[sign] = numm
            # 需要删除hji这个内部节点
            # 首选得到这个内部节点所有的
            for hco in self.xiashu_leaf(dd):
                if hco in coppdict:
                    del coppdict[hco]
            # 树加1
            sign += 1
            self.noderela = coppdict.copy()
            tree_set[sign] = self.noderela.copy()
        return tree_set, adict

    # 计算正确率的函数
    def compuer_correct(self, exli_real, exli_pre):
        if len(exli_pre) == 0:
            return 0
        else:
            corr = np.array(exli_pre)[np.array(exli_pre) == np.array(exli_real)]
            return len(corr) / len(exli_pre)

    # 交叉验证函数
    def jiaocha_tree(self, treeset):  #输出最终的树
        # 正确率的字典
        correct = {}

        # 遍历树的集合
        for jj in treeset:
            self.noderela = treeset[jj]
            yuce = self.pre_tree(self.test_dtdata)
            # 真实的预测值
            real = self.test_dtdata[:, -1]
            # 计算正确率
            correct[jj] = self.compuer_correct(real, yuce)
        # 获得最大的，如果有相同的，获取数目最小的键
        num = 0
        leys = ''
        for jj in correct:
            if correct[jj] > num:
                num = correct[jj]
                leys = jj
            elif num == correct[jj]:
                if jj < leys:
                    leys = jj
        return treeset[leys], num


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

        uu = DT(tree_length=shendu)
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
    plt.plot(list(range(2, 13)), xunliande, 'o--', label='训练', lw=2)
    plt.plot(list(range(2, 13)), yazhengde, '*--', label='验证', lw=2)
    plt.plot(list(range(2, 13)), yucede, 's--', label='预测', lw=2)
    plt.xlabel('树的初始深度')
    plt.xlim(1, 14)

    plt.ylabel('正确率')
    plt.legend(shadow=True, fancybox=True)
    plt.show()