# -*- coding：utf-8 -*-
# &Author  AnFany

# 两层的Stacking分类


#  第一层6个模型：随机森林，AdaBoost，GBDT，LightGBM，XGBoost，CatBoost
#  第二层模型：BP神经网络分类

# 引入数据文件
import adult_Stacking_Data as adult

# 引入绘图库包
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 中文字体名称
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
# 设置正确率的刻度与子刻度
y_toge = MultipleLocator(0.02)  # 将y轴主刻度标签设置为0.1的倍数
y_son = MultipleLocator(0.01)  # 将此y轴次刻度标签设置为0.01的倍数
# 引入需要用到的模型的库包
# 随机森林
from sklearn.ensemble import RandomForestClassifier as RF
# AdaBoost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
# GBDT
from sklearn.ensemble import GradientBoostingClassifier
# XGBoost
import xgboost as xgb
# LightGBM
import lightgbm as lgbm
# CatBoost
import catboost as cb
# BP神经网络分类
import tensorflow as tf
import bp_Classify as bp

# 其他库包
import numpy as np
import pandas as pd
from collections import OrderedDict  # python字典是无序的，此包是有序的
# 格式化输出混淆矩阵
from prettytable import PrettyTable as PT
'''
第一部分：数据处理模型
'''

class DATA:

    def __init__(self, datadict=adult.data_dict, mubiao='Money'):
        self.data = datadict
        self.k = 8  # 因为需要对每个模型进行单独的数据处理，因此这个折数对于每个模型都必须是一样的

        # 训练数据
        self.chidata = self.data['train']

        # 预测数据
        self.nodata = self.data['predict']

        # 类别型数据的编号
        self.catsign = self.Sign()

        # 目标字段
        self.ziduan = mubiao

        # 对于归一化化和标准化的处理，要记录转化的值，在这里将2者统一。反处理时，预测值需要乘以self.fenmu 加上self.cha
        # 分类问题不涉及反处理
        # 对于CatBoost而言，因为需要将训练数据的字段，转为数字，因此结果中需要将这个数字转化为真实的类别名称
        self.typedict = self.TransType()

        # 本程序第二层采用BP神经网络，为了适用于多分类的情形，需要获得类别数
        self.typecount = self.Getcount()

    # 计算类别数
    def Getcount(self):
        nulist = list(set(list(self.chidata[self.ziduan])))
        return len(nulist)

    # 因为对于CatBoost而言，不需要进行类别型特征的处理，但是需要类别型特征的标号
    def Sign(self):
        sign = []
        numlist = self.chidata.values[0][: -1]  # 不包括最后的目标字段
        for jj in range(len(numlist)):
            try:
                numlist[jj] + 9
            except TypeError:
                sign.append(jj)
        return sign
    # 对于CatBoost而言，需要对目标字段进行数字化处理，
    def TransType(self):
        tdict = {}
        nulist = sorted(list(set(list(self.chidata[self.ziduan]))))
        for jj in nulist:
            tdict[jj] = nulist.index(jj)
        return tdict


    # 将目标字段转化为数字(CatBoost)
    def TargetoDi(self):
        # 将目标字段按照字典的形式转变为数字
        self.chidata[self.ziduan] = [self.typedict[jj] for jj in self.chidata[self.ziduan]]
        return print('CatBoost目标字段转化完毕')

    # 因为引入的BP分类模型，输出数据需要经过独热化处理
    def OneH(self):
        # 首先定义一个全0数组
        zero = [0] * self.typecount
        # 定义一个类别序列
        typelist = zero.copy()
        for jj in self.typedict:
            typelist[self.typedict[jj]] = jj

        # 开始更改目标字段的数据
        newdata  = []
        for jj in self.chidata[self.ziduan]:
            ss = zero.copy()
            ss[typelist.index(jj)] = 1
            newdata.append(ss)

        self.chidata[self.ziduan] = newdata

        return '目标字段独热化完毕'


    # 类别型特征数字标签化函数，
    def CAtoDI(self):
        # 如果特征值不能执行数字加法运算，则视为类别型特征
        for tezheng in self.chidata:
            try:
                self.chidata[tezheng].values[0] + 1
            except TypeError:
                numlist = sorted(list(set(list(self.chidata[tezheng]))))
                self.chidata[tezheng] = [numlist.index(hh) for hh in self.chidata[tezheng]]
                try:
                    self.nodata[tezheng] = [numlist.index(ss) for ss in self.nodata[tezheng]]
                except ValueError:
                    print('特征%s：预测比训练的多了值' % (tezheng))
        return print('数字化处理完毕')


    # 归一化函数
    def Normal(self):
        # 在此之前需要把类别型标签去掉,否则会报错
        for tezheng in self.chidata:
            maxnum = max(list(self.chidata[tezheng]))
            minum = min(list(self.chidata[tezheng]))
            if maxnum == minum:
                self.chidata[tezheng] = [1 for hh in self.chidata[tezheng]]
                self.nodata[tezheng] = [1 for ss in self.nodata[tezheng]]
            else:
                self.chidata[tezheng] = [(hh - minum) / (maxnum - minum) for hh in self.chidata[tezheng]]
                self.nodata[tezheng] = [(ss - minum) / (maxnum - minum) for ss in self.nodata[tezheng]]
        return print('归一化处理完毕')

    # 标准化函数
    def Stand(self):
        # 在此之前需要把类别型标签去掉,否则会报错
        for tezheng in self.chidata:
            standnum = np.std(np.array(list(self.chidata[tezheng])), ddof=1)  # 计算有偏的标准差
            meanum = np.mean(np.array(list(self.chidata[tezheng])))

            if meanum == 0:
                self.chidata[tezheng] = [1 for hh in self.chidata[tezheng]]
                self.nodata[tezheng] = [1 for ss in self.nodata[tezheng]]
            else:
                self.chidata[tezheng] = [(hh - standnum) / meanum for hh in self.chidata[tezheng]]
                self.nodata[tezheng] = [(ss - standnum) / meanum for ss in self.nodata[tezheng]]
        return print('标准化处理完毕')

    # 定义Kfold的函数，也就是将原始的训练数据集分为k对训练数据和验证数据的组合
    def Kfold(self):
        # 因为每个模型需要将验证数据结合的结果集成起来，为了方便起见，在这里固定每一折数中的数据集合
        datanum = self.chidata.values
        # 数据集总长度
        length = len(datanum)
        alist = np.arange(length)
        np.random.seed(1990)
        np.random.shuffle(alist)  # 随机打乱数据对BPNN,SVM而言是有益处的，而对于决策树之类的模型而言没有影响

        # 验证数据的长度
        yanlem = int(length / self.k)

        # 存储数据集的字典
        datai = {}
        datai['predict'] = self.nodata.values

        # 开始处理Kfold
        for kk in range(self.k):
            datai[kk] = OrderedDict()
            if kk == 0:
                datai[kk]['train'] = datanum[alist[(kk + 1) * yanlem:]]
                datai[kk]['test'] = datanum[alist[: (kk + 1) * yanlem]]
            elif kk == self.k - 1:
                datai[kk]['train'] = datanum[alist[: kk * yanlem]]
                datai[kk]['test'] = datanum[alist[kk * yanlem:]]
            else:
                datai[kk]['test'] = datanum[alist[kk * yanlem: (kk + 1) * yanlem]]
                signlist = list(alist[: kk * yanlem]) + list(alist[(kk + 1) * yanlem:])
                datai[kk]['train'] = datanum[signlist]
        # 返回的数据集形式{0：{'train':data, 'test':data}，……，self.k-1:{'train':data, 'test':data}, 'predict':data}
        print('K折处理完毕')
        return datai


'''
第二部分：第一层的模型运行阶段
'''
# 可以任意添加模型
class MODELONE:

    def __init__(self, exdict, zidan='Money'):

        # 验证数据集的预测结果
        self.yanzhneg_pr = []

        # 验证数据集的真实结果
        self.yanzhneg_real = []

        # 预测数据集的预测结果
        self.predi = []

        # 预测数据集的真实姐夫哦
        self.preal = []

        # 目标字段名称
        self.zi = zidan

        # 数据结构和数据处理类的保持一致，要把验证数据集的输入和真实的输出合二为一
        self.datai = {}

        # 记录每个模型最终误差的字典
        self.error_dict = OrderedDict()

        # 数字类别与正常类别的对应字典
        self.tydict = exdict

    # 将第一层的结果转换为何数据结构处理类中一样的数据结构的函数
    #  也就是{'train':dataframe, 'predict':dataframe}样式的字典
    def DataStru(self):
        self.datai['train'] = np.row_stack((np.array(self.yanzhneg_pr), np.array(self.yanzhneg_real)))  # 此处添加行
        self.datai['predict'] = np.row_stack((np.array(self.predi), np.array(self.preal)))
        # 将训练数据转置
        datapst = self.datai['train'].T
        # 为训练数据定义DataFrame的列名
        mingcheng = ['第%s个模型列' % str(dd) for dd in list(range(len(self.datai['train']) - 1))] + [self.zi]
        self.datai['train'] = pd.DataFrame(datapst, columns=mingcheng)

        # 将预测数据转置
        dapst = self.datai['predict'].T
        # 为训练数据定义DataFrame的列名
        mingche= ['第%s个模型列' % str(dd) for dd in list(range(len(self.datai['predict']) - 1))] + [self.zi]
        self.datai['predict'] = pd.DataFrame(dapst, columns=mingche)
        return print('二层的数据准备完毕')

    # 创建将预测的多维的数字类别转化为一维原始名称类别的函数
    def AntiTae(self, relist):
        huhuan = {self.tydict[ll]: ll for ll in self.tydict}
        return [huhuan[dd] for dd in relist]

    # 创建将预测的多维的数字类别转化为一维原始名称类别的函数
    def MTae(self, relist):
        yiwellist = []
        for jj in relist:
            yiwellist.append(list(jj).index(max(list(jj))))
        # 首先将字典键值互换
        huhuan = {self.tydict[ll]: ll for ll in self.tydict}
        return [huhuan[dd] for dd in yiwellist]

    # 混淆矩阵的函数
    def Tom(self, reallist, prelist):
        '''
        :param reallist: 真实的类别列表
        :param prelist:  预测的类别列表
        :return: 每个类别预测为所有类别的个数字典
        '''
        coundict = {}
        for jj in list(set(reallist)):
            coundict[jj] = {}
            for hh in list(set(reallist)):
                coundict[jj][hh] = len([i for i, j in zip(reallist, prelist) if i == jj and j == hh])
        return coundict

    # 定义输出混淆矩阵的函数
    def ConfuseMatrix(self, reallist, prelist):
        '''
        :param reallist: 真实的类别列表
        :param prelist: 预测的类别列表
        :return: 输出混淆矩阵
        '''
        zidian = self.Tom(reallist, prelist)
        lieming = sorted(zidian.keys())
        table = PT(['混淆矩阵'] + ['预测%s' % d for d in lieming])
        for jj in lieming:
            table.add_row(['实际%s' % jj] + [zidian[jj][kk] for kk in lieming])
        return table

    # 定义计算F1度量的函数
    def F1(self, realist, prelist):
        '''
        :param realist: 真实的类别列表
        :param prelist: 预测的类别列表
        :return: F1度量
        '''
        condict = self.Tom(realist, prelist)
        zongshu = 0
        zhengque = 0
        zhao_cu = []  # 存储每个类别的召回率
        for cu in condict:
            zq = 0
            zs = 0
            for hh in condict[cu]:
                geshu = condict[cu][hh]
                if cu == hh:
                    zhengque += geshu
                    zq = geshu
                zongshu += geshu
                zs += geshu
            zhao_cu.append(zq / zs)
        # 计算精确率
        jingque = zhengque / zongshu
        # 计算类别召回率
        zhaohui = np.mean(np.array(zhao_cu))
        # f1度量
        f_degree = 2 * jingque * zhaohui / (jingque + zhaohui)
        return f_degree

    # 随机森林
    def RF_First(self, data, n_estimators=1000, max_features='sqrt'):
        # 存储每一折中验证数据集的预测结果
        yanzhenglist = []
        # 存储每一折中验证数据集的真实结果
        yanzhenglist_real = []
        # 存储每一折中预测数据集的预测结果
        prelist = []

        # 存储训练、验证、预测数据的误差
        errorlsit = []
        # 开始每一折的训练,因为这个折数的字典是有序的,因此不用考虑每一折的顺序。
        for zhe in [zheshu for zheshu in data.keys() if zheshu != 'predict']:
            model = RF(n_estimators=n_estimators, max_features=max_features)
            model.fit(data[zhe]['train'][:, :-1], data[zhe]['train'][:, -1])
            # 注意存储验证数据集结果和预测数据集结果的不同
            # 训练数据集的预测结果
            xul = model.predict(data[zhe]['train'][:, :-1])
            # 验证的预测结果
            yanre = model.predict(data[zhe]['test'][:, :-1])
            #预测的预测结果
            prer = model.predict(data['predict'][:, :-1])

            yanzhenglist += list(yanre)
            yanzhenglist_real += list(data[zhe]['test'][:, -1])
            prelist.append(prer)
            # 每计算一折后，要计算训练、验证、预测数据的误差
            xx = self.F1(xul, data[zhe]['train'][:, -1])

            yy = self.F1(yanre, data[zhe]['test'][:, -1])

            pp = self.F1(prer, data['predict'][:, -1])

            errorlsit.append([xx, yy, pp])
        # 针对预测数据集的预测结果计算均值
        meanPre = np.mean(np.array(prelist), axis=0)
        # 开始结合
        self.yanzhneg_pr.append(yanzhenglist)
        self.yanzhneg_real = yanzhenglist_real
        self.predi.append(meanPre)
        self.preal = data['predict'][:, -1]

        # 储存误差
        self.error_dict['随机森林'] = np.mean(np.array(errorlsit), axis=0)
        return print('1层中的随机森林运行完毕')

    # AdaBoost
    def Adaboost_First(self, data, max_depth=5, n_estimators=1000):
        # 存储每一折中验证数据集的预测结果
        yanzhenglist = []
        # 存储每一折中验证数据集的真实结果
        yanzhenglist_real = []
        # 存储每一折中预测数据集的预测结果
        prelist = []

        # 存储训练、验证、预测数据的误差
        errorlsit = []

        # 开始每一折的训练
        for zhe in [zheshu for zheshu in data.keys() if zheshu != 'predict']:
            model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth),
                                       algorithm="SAMME",
                                       n_estimators=n_estimators, learning_rate=0.8)
            model.fit(data[zhe]['train'][:, :-1], data[zhe]['train'][:, -1])
            # 注意存储验证数据集结果和预测数据集结果的不同
            # 训练数据集的预测结果
            xul = model.predict(data[zhe]['train'][:, :-1])
            # 验证的预测结果
            yanre = model.predict(data[zhe]['test'][:, :-1])
            #预测的预测结果
            prer = model.predict(data['predict'][:, :-1])
            yanzhenglist += list(yanre)
            yanzhenglist_real += list(data[zhe]['test'][:, -1])
            prelist.append(prer)

            # 每计算一折后，要计算训练、验证、预测数据的误差
            xx = self.F1(xul, data[zhe]['train'][:, -1])

            yy = self.F1(yanre, data[zhe]['test'][:, -1])

            pp = self.F1(prer, data['predict'][:, -1])

            errorlsit.append([xx, yy, pp])

        # 针对预测数据集的预测结果计算均值
        meanPre = np.mean(np.array(prelist), axis=0)
        # 开始结合
        self.yanzhneg_pr.append(yanzhenglist)
        self.yanzhneg_real = yanzhenglist_real
        self.predi.append(meanPre)
        self.preal = data['predict'][:, -1]
        # 储存误差
        self.error_dict['AdaBoost'] = np.mean(np.array(errorlsit), axis=0)

        return print('1层中的AdaBoost运行完毕')

    # GBDT
    def GBDT_First(self, data, max_depth=5, n_estimators=280):
        # 存储每一折中验证数据集的预测结果
        yanzhenglist = []
        # 存储每一折中验证数据集的真实结果
        yanzhenglist_real = []
        # 存储每一折中预测数据集的预测结果
        prelist = []

        # 存储训练、验证、预测数据的误差
        errorlsit = []
        # 开始每一折的训练
        for zhe in [zheshu for zheshu in data.keys() if zheshu != 'predict']:
            model = GradientBoostingClassifier(loss='deviance', n_estimators=n_estimators, max_depth=max_depth,
                                               learning_rate=0.1, max_features='sqrt')
            model.fit(data[zhe]['train'][:, :-1], data[zhe]['train'][:, -1])
            # 注意存储验证数据集结果和预测数据集结果的不同
            # 训练数据集的预测结果
            xul = model.predict(data[zhe]['train'][:, :-1])
            # 验证的预测结果
            yanre = model.predict(data[zhe]['test'][:, :-1])
            # 预测的预测结果
            prer = model.predict(data['predict'][:, :-1])
            yanzhenglist += list(yanre)
            yanzhenglist_real += list(data[zhe]['test'][:, -1])
            prelist.append(prer)

            # 每计算一折后，要计算训练、验证、预测数据的误差
            xx = self.F1(xul, data[zhe]['train'][:, -1])

            yy = self.F1(yanre, data[zhe]['test'][:, -1])

            pp = self.F1(prer, data['predict'][:, -1])

            errorlsit.append([xx, yy, pp])

        # 针对预测数据集的预测结果计算均值
        meanPre = np.mean(np.array(prelist), axis=0)
        # 开始结合
        self.yanzhneg_pr.append(yanzhenglist)
        self.yanzhneg_real = yanzhenglist_real
        self.predi.append(meanPre)
        self.preal = data['predict'][:, -1]
        # 储存误差
        self.error_dict['GBDT'] = np.mean(np.array(errorlsit), axis=0)

        return print('1层中的GBDT运行完毕')

    # LightGBM
    def LightGBM_First(self, data, max_depth=5, n_estimators=235):
        # 存储每一折中验证数据集的预测结果
        yanzhenglist = []
        # 存储每一折中验证数据集的真实结果
        yanzhenglist_real = []
        # 存储每一折中预测数据集的预测结果
        prelist = []

        # 存储训练、验证、预测数据的误差
        errorlsit = []
        # 开始每一折的训练
        for zhe in [zheshu for zheshu in data.keys() if zheshu != 'predict']:
            model = lgbm.LGBMClassifier(boosting_type='gbdt', objective='binary', num_leaves=50,
                                        learning_rate=0.1, n_estimators=n_estimators, max_depth=max_depth,
                                        bagging_fraction=0.9, feature_fraction=0.9, reg_lambda=0.2)

            model.fit(data[zhe]['train'][:, :-1], data[zhe]['train'][:, -1])
            # 注意存储验证数据集结果和预测数据集结果的不同
            # 训练数据集的预测结果
            xul = model.predict(data[zhe]['train'][:, :-1])
            # 验证的预测结果
            yanre = model.predict(data[zhe]['test'][:, :-1])
            # 预测的预测结果
            prer = model.predict(data['predict'][:, :-1])
            yanzhenglist += list(yanre)
            yanzhenglist_real += list(data[zhe]['test'][:, -1])
            prelist.append(prer)
            # 每计算一折后，要计算训练、验证、预测数据的误差
            xx = self.F1(xul, data[zhe]['train'][:, -1])
            yy = self.F1(yanre, data[zhe]['test'][:, -1])
            pp = self.F1(prer, data['predict'][:, -1])
            errorlsit.append([xx, yy, pp])
        # 针对预测数据集的预测结果计算均值
        meanPre = np.mean(np.array(prelist), axis=0)
        # 开始结合
        self.yanzhneg_pr.append(yanzhenglist)
        self.yanzhneg_real = yanzhenglist_real
        self.predi.append(meanPre)
        self.preal = data['predict'][:, -1]
        # 储存误差
        self.error_dict['LightGBM'] = np.mean(np.array(errorlsit), axis=0)

        return print('1层中的LightGBM运行完毕')

    # XGBoost
    def XGBoost_First(self, data, max_depth=5, n_estimators=320):
        # 存储每一折中验证数据集的预测结果
        yanzhenglist = []
        # 存储每一折中验证数据集的真实结果
        yanzhenglist_real = []
        # 存储每一折中预测数据集的预测结果
        prelist = []
        # 存储训练、验证、预测数据的误差
        errorlsit = []
        # 开始每一折的训练
        for zhe in [zheshu for zheshu in data.keys() if zheshu != 'predict']:
            model = xgb.XGBClassifier(max_depth=max_depth, learning_rate=0.1, n_estimators=n_estimators,
                                      silent=True, objective='binary:logistic', booster='gbtree')
            model.fit(data[zhe]['train'][:, :-1], data[zhe]['train'][:, -1])
            # 注意存储验证数据集结果和预测数据集结果的不同
            # 训练数据集的预测结果
            xul = model.predict(data[zhe]['train'][:, :-1])
            # 验证的预测结果
            yanre = model.predict(data[zhe]['test'][:, :-1])
            # 预测的预测结果
            prer = model.predict(data['predict'][:, :-1])
            yanzhenglist += list(yanre)
            yanzhenglist_real += list(data[zhe]['test'][:, -1])
            prelist.append(prer)
            # 每计算一折后，要计算训练、验证、预测数据的误差
            xx = self.F1(xul, data[zhe]['train'][:, -1])
            yy = self.F1(yanre, data[zhe]['test'][:, -1])
            pp = self.F1(prer, data['predict'][:, -1])
            errorlsit.append([xx, yy, pp])
        # 针对预测数据集的预测结果计算均值
        meanPre = np.mean(np.array(prelist), axis=0)
        # 开始结合
        self.yanzhneg_pr.append(yanzhenglist)
        self.yanzhneg_real = yanzhenglist_real
        self.predi.append(meanPre)
        self.preal = data['predict'][:, -1]
        # 储存误差
        self.error_dict['XGBoost'] = np.mean(np.array(errorlsit), axis=0)
        return print('1层中的XGBoost运行完毕')

    # CatBoost
    def CatBoost_First(self, data, catsign, depth=7, iterations=200):

        # 存储每一折中验证数据集的预测结果
        yanzhenglist = []
        # 存储每一折中验证数据集的真实结果
        yanzhenglist_real = []
        # 存储每一折中预测数据集的预测结果
        prelist = []
        # 存储训练、验证、预测数据的误差
        errorlsit = []
        # 开始每一折的训练
        for zhe in [zheshu for zheshu in data.keys() if zheshu != 'predict']:
            model = cb.CatBoostClassifier(iterations=iterations, depth=depth, learning_rate=0.5,
                                          loss_function='Logloss',
                                          logging_level='Verbose')

            model.fit(data[zhe]['train'][:, :-1], data[zhe]['train'][:, -1], cat_features=catsign)
            # 注意存储验证数据集结果和预测数据集结果的不同
            # 训练数据集的预测结果
            xul = model.predict(data[zhe]['train'][:, :-1])
            # 验证的预测结果
            yanre = model.predict(data[zhe]['test'][:, :-1])
            # 预测的预测结果
            prer = model.predict(data['predict'][:, :-1])
            yanzhenglist += list(yanre)
            yanzhenglist_real += list(data[zhe]['test'][:, -1])
            prelist.append(prer)
            # 每计算一折后，要计算训练、验证、预测数据的误差
            xx = self.F1(xul, data[zhe]['train'][:, -1])
            yy = self.F1(yanre, data[zhe]['test'][:, -1])
            pp = self.F1(prer, data['predict'][:, -1])
            errorlsit.append([xx, yy, pp])
        # 针对预测数据集的预测结果计算均值
        meanPre = np.mean(np.array(prelist), axis=0)
        # 开始结合
        self.yanzhneg_pr.append(yanzhenglist)
        self.yanzhneg_real = yanzhenglist_real
        self.predi.append(meanPre)
        self.preal = data['predict'][:, -1]
        # 储存误差
        self.error_dict['CatBoost'] = np.mean(np.array(errorlsit), axis=0)
        return print('1层中的CatBoost运行完毕')

'''
第三部分：第二层的模型运行阶段 可以任意更换模型
'''
class MODETWO:

    def __init__(self, in_tr_data, out_tr_data, in_pre_data, out_pre):
        self.xdata = in_tr_data
        self.ydata = out_tr_data

        self.xdatapre = in_pre_data
        self.ydapre = out_pre



    # BP神经网络回归
    def BP(self, hiddenlayers=3, hiddennodes=100, learn_rate=0.02,
           itertimes=30000, batch_size=50, activate_func='tanh'):
        *guocheng, zuiyou, hiss = bp.Ten_train(self.xdata, self.ydata, self.xdatapre,
                                               self.ydapre, hiddenlayers=hiddenlayers,
                                               hiddennodes=hiddennodes, learn_rate=learn_rate,
                                               itertimes=itertimes, batch_size=batch_size, activate_func=activate_func)

        return guocheng, zuiyou, hiss

'''
第四部分：绘制图，绘制第一层各个模型中训练，验证数据的误差，
以及最终的预测数据的真实值和误差值的对比
'''
# 定义绘制第一层模型训练、验证、预测数据的F1度量的函数
# 根据字典绘制不同参数下评分的对比柱状图
def Plot_RMSE_ONE_Stacking(exdict, kaudu=0.2):
    '''
    :param exdict: 不同模型F1度量
    :return: 柱状图
    '''
    # 参数组合列表
    palist = exdict.keys()
    # 对应的训练数据的评分
    trsore = [exdict[hh][0] for hh in palist]
    # 对应的测试数据的评分
    tesore = [exdict[hh][1] for hh in palist]
    # 对应的预测数据的评分
    presore = [exdict[hh][2] for hh in palist]

    # 开始绘制柱状图
    fig, ax = plt.subplots()
    # 柱的个数
    ind = np.array(list(range(len(trsore))))
    # 绘制柱状
    ax.bar(ind - kaudu, trsore, kaudu, color='SkyBlue', label='训练')
    ax.bar(ind, tesore, kaudu, color='IndianRed', label='测试')
    ax.bar(ind + kaudu, presore, kaudu, color='slateblue', label='预测')
    # xy轴的标签
    ax.set_ylabel('召回率')
    ax.set_xlabel('Stacking第一层中的模型')
    # 设置刻度
    ax.set_xticks(ind)
    ax.set_xticklabels(palist)

    leg = ax.legend(loc='best', ncol=3, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.8)

    plt.title('Stacking第一层中模型的召回率')
    plt.savefig(r'C:\Users\GWT9\Desktop\Stacking_adult.jpg')

    return print('一层不同模型对比')


# 绘制每一次迭代过程中的训练、验证的误差以及正确率

def plotcurve(bpnn):
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
    plt.title('训练VS验证结果对比', fontsize=16)

    plt.savefig(r'C:\Users\GWT9\Desktop\stacking_guo.jpg')


    return print('过程绘图完毕')

'''
第五部分：Stacking主函数
'''

if __name__ == "__main__":
    #  第一层6个模型：随机森林，AdaBoost，GBDT，LightGBM，XGBoost，CatBoost

    # 下面依次为每个模型建立数据
    # 随机森林、AdaBoost，GBDT，LIghtGNM，XGBoost都是一样的
    rf_data = DATA()
    rf_data.CAtoDI()  # 标签数字化
    data_rf = rf_data.Kfold()  # 折数


    # CatBoost
    cat_data = DATA()  # 不用处理
    cat_data.TargetoDi()   # 需要将目标字段数字化
    data_cat = cat_data.Kfold()  # 折数


    # 开始建立Stacking第一层的模型
    one_stacking = MODELONE(exdict=rf_data.typedict)
    # 随机森林
    one_stacking.RF_First(data_rf)
    # AdaBoost
    one_stacking.Adaboost_First(data_rf)
    # GBDT
    one_stacking.GBDT_First(data_rf)
    # LightGBM
    one_stacking.LightGBM_First(data_rf)
    # XGBoost
    one_stacking.XGBoost_First(data_rf)
    # CatBoost
    one_stacking.CatBoost_First(data_cat, cat_data.catsign)

    # 第二层的数据准备
    one_stacking.DataStru()
    data_two = one_stacking.datai

    # 第二层的数据处理
    erce_data = DATA(datadict=data_two)
    erce_data.CAtoDI()  # 因为输出的都是类别，因此要标签化
    erce_data.Normal()
    erce_data.OneH()  # 训练的输出独热化处理
    # 为了获得更好的模型，在这里设置验证数据，
    bpdatadict = erce_data.Kfold()  # 为了简便，不再进行交叉验证获得最佳的参数

    # 第二层建模,
    stacking_two = MODETWO(bpdatadict[0]['train'][:, :-1],
                           np.array(list(bpdatadict[0]['train'][:, -1])),
                           bpdatadict[0]['test'][:, :-1],
                           np.array(list(bpdatadict[0]['test'][:, -1])))

    # 训练的输出值，预测的输出值, 每一次迭代训练和预测的误差
    error_acc, signi, gir = stacking_two.BP()

    # 训练完成后读取最优的参数，在计算最终的预测结果
    graph = tf.train.import_meta_graph(r'E:\tensorflow_Learn\Stacking\adult\model-%s.meta' % signi)
    ses = tf.Session()
    graph.restore(ses, tf.train.latest_checkpoint(r'E:\tensorflow_Learn\Stacking\adult'))
    op_to_restore = tf.get_default_graph().get_tensor_by_name("Add_%s:0" % gir)
    w1 = tf.get_default_graph().get_tensor_by_name("x_data:0")
    feed_dict = {w1: bpdatadict['predict'][:, :-1]}
    dgsio = ses.run(op_to_restore, feed_dict)

    #  将输出的结果转变为数字化的类别，然后再转化为真实的类别，输出混淆矩阵
    bp_out_type = one_stacking.MTae(bp.judge(dgsio))
    bp_real_type = one_stacking.AntiTae(bpdatadict['predict'][:, -1])

    # 绘制第一层中各个模型的误差图
    Plot_RMSE_ONE_Stacking(one_stacking.error_dict)
    # 绘制第二层模型中的训练和预测误差
    plotcurve(error_acc)

    fru = one_stacking.ConfuseMatrix(bp_real_type, bp_out_type)



