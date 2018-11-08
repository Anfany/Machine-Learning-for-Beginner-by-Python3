# -*- coding：utf-8 -*-
# &Author  AnFany

# '''CART回归树：可处理连续、离散的变量，不支持输出数据是多维的。  可以利用多个决策树解决多维输出的问题
# 测试数据和训练数据的字段顺序必须一样，因为本程序在设定规则按的是字段的编号，而不是名字
# 回归树，因为精确度问题。不运行剪枝这一环节，当训练数据，预测数据和验证数据的MSE不再降低时，确定最佳的树的深度
# 回归树首先要降低偏差问题，偏差只有小到一定程度，再降低方差。训练数据的精确度可看作降低偏差，而预测数据和验证数据的精确度可看作降低方差
# '''

# print(__doc__)
# 引入数据
import Data_DT_Regression as dtda
import copy
import numpy as np

# 定义函数
class DT:
    def __init__(self, train_dtdata=dtda.dt_data, pre_dtdata=dtda.test_data, tree_length=5):

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

        # 避免树过大，采用限制书的深度
        self.tree_length = tree_length


    # #  根据类别的数组计算基尼指数
    # def jini_zhishu(self, exlist):
    #     dnum = 0
    #     leng = len(exlist)
    #     for hh in list(set(exlist)):
    #         dnum += (list(exlist).count(hh) / leng) ** 2
    #     return 1 - dnum


    #  根据类别的数组计算数组间的标准差
    def biaozhuncha(self, exlist):
        exlist = np.array(exlist)
        if len(exlist) <= 1:
            return 0
        else:
            return np.std(exlist, ddof=1)  # ddof=1计算样本的标准差

    #  计算标准差的函数
    def jini_xishu(self, tezheng, leibie):  # 输入特征数据，类别数据，返回最小标准差对应的值
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
            # 开始遍历每一个中间值，计算对应的标准差
            length = len(leibie)
            # 存储标准差的值
            save_ji, jini = np.inf, 0
            number = ''
            for mi in midd:
                #  计算标准差和
                onelist = leibie[tezheng <= mi]
                twolist = leibie[tezheng > mi]
                jini = (len(onelist) / length) * self.biaozhuncha(onelist) + (len(twolist) / length) * self.biaozhuncha(twolist)
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
            # 开始遍历每一个值，计算对应的标准差
            length = len(leibie)
            # 存储标准差的值
            jini, save_ji = 0, np.inf
            number = ''
            for mi in quzhong:
                #  计算标准差和
                onelist = leibie[tezheng == mi]
                twolist = leibie[tezheng != mi]
                jini = (len(onelist) / length) * self.biaozhuncha(onelist) + (len(twolist) / length) * self.biaozhuncha(
                     twolist)
                if jini <= save_ji:
                    save_ji = jini
                    number = mi
            return number, save_ji  # 该特征最好的分割值，以及该特征最小的标准差


    # 数据集确定分类特征以及属性的函数
    def feature_zhi(self, datadist):  # 输入的数据集字典，输出最优的特征编号，以及对应的值，还有标准差
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
            return tezhengsign, jini, number  # 特征编号, 该特征最好的分割值，该数据集最小的标准差
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

            print('所有节点的个数：', len(self.fenlei_shujuji))
            print('需要分裂的数据集的个数：', len(self.node_shujuji))

        return 'done'

    # 根据树得出每一个节点数据集的结果
    def jieguo_tree(self):
        # 根据每一个数据得到每一个节点对应的结果
        shujuji_jieguo = {}
        for shuju in self.node_shujuji:
            zuihang = self.node_shujuji[shuju][:, -1]
            #  选择均值
            shujuji_jieguo[shuju] = round(np.mean(np.array(zuihang)), 1)

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


    # 判读数据是否符合这个规则的函数
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


    # 计算每一个节点的剪枝的标准差
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

    # 计算每一个内部节点的下属叶子节点的标准差之和
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
            # 存储内部节点剪枝标准差的字典
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


    #  计算预测值和真实值误差平方和
    def compuer_mse(self, exli_real, exli_pre):
        if len(exli_pre) == 0:
            return 0
        else:
            exli_pre = np.array(exli_pre)
            exli_real = np.array(exli_real)
            corr = exli_pre - exli_real
            return np.sum(np.array([i ** 2 for i in corr])) / len(corr)

    # 交叉验证函数
    def jiaocha_tree(self, treeset):  #输出最终的树
        # 最小MSE的字典
        correct = {}

        # 遍历树的集合
        for jj in treeset:
            self.noderela = treeset[jj]
            yuce = self.pre_tree(self.test_dtdata)
            # 真实的预测值
            real = self.test_dtdata[:, -1]
            # 计算MSE
            correct[jj] = self.compuer_mse(real, yuce)
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


#  数据量较大，折线图对比不容易看清，训练数据随机选取200条，验证和预测随机选取100条展示
def selet(prdata, reda, count=200):
    if len(reda) <= count:
        return prdata, reda
    fu = np.arange(len(reda))

    du = np.random.choice(fu, count)

    return np.array(prdata)[du], np.array(reda)[du]



# 根据不同的深度。看精确率的变化
if __name__ == '__main__':
    # 根据树的不同的初始深度，看正确率的变化
    xunliande = []
    yazhengde = []
    yucede = []

    for shendu in range(2, 25):

        uu = DT(tree_length=shendu)
        # 完全成长的树
        uu.grow_tree()
        #  不再保留剪枝的过程
        # # 剪枝形成的树的集
        # gu = uu.prue_tree()
        # # 交叉验证形成的最好的树
        # cc = uu.jiaocha_tree(gu[0])
        # # 根据最好的树预测新的数据集的结果
        # uu.noderela = cc[0]

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
    reuu = DT(tree_length=zuiyoushendu)
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
    a, b = selet(trainnum, reuu.train_dtdata[:, -1])
    plt.plot(list(range(len(a))), a, list(range(len(b))), b)

    plt.legend(['预测', '真实'])
    plt.title('训练数据')


    plt.subplot(223)
    c, d = selet(yannum, reuu.test_dtdata[:, -1], count=100)
    plt.plot(list(range(len(c))), c, list(range(len(d))), d)
    plt.legend(['预测', '真实'])
    plt.title('验证数据')

    plt.subplot(224)
    e, f = selet(prenum, reuu.pre_dtdata[:, -1], count=100)
    plt.plot(list(range(len(e))), c, list(range(len(d))), f)
    plt.legend(['预测', '真实'])
    plt.title('预测数据')

    plt.show()