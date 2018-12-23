# -*- coding：utf-8 -*-
# &Author  AnFany

# 两层的Stacking回归
from wxpy import *
bot = Bot(cache_path=True)

#  第一层6个模型：随机森林，AdaBoost，GBDT，LightGBM，XGBoost，CatBoost
#  第二层模型：BP神经网络回归

# 引入数据文件
import pm25_Stacking_Data as pm25

# 引入绘图库包
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号

# 引入需要用到的模型的库包
# 随机森林
from sklearn.ensemble import RandomForestRegressor as RF
# AdaBoost
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
# GBDT
from sklearn.ensemble import GradientBoostingRegressor
# XGBoost
import xgboost as xgb
# LightGBM
import lightgbm as lgbm
# CatBoost
import catboost as cb
# BP神经网络回归
import BP_Regression as bp

# 其他库包
import tensorflow as tf
import numpy as np
import pandas as pd
from collections import OrderedDict  # python字典是无序的，此包是有序的
import os
os.chdir(r'E:\tensorflow_Learn\Stacking\pm25')
'''
第一部分：数据处理模型
'''

class DATA:

    def __init__(self, datadict=pm25.data_dict, mubiao='pm2.5'):
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
        self.fenmu = 1  # 相当于标准化的标准差, 归一化的最大值减去最小值
        self.cha = 0   # 相当于标准化的平均值, 归一化的最小值

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

    # 对于归一化和标准化的函数，要记录目标字段的转化值，因为要进行反归一化

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
                if tezheng == self.ziduan:
                    self.fenmu = maxnum - minum
                    self.cha = minum
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
                if tezheng == self.ziduan:
                    self.fenmu = standnum
                    self.cha = meanum
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

    def __init__(self, zidan='pm2.5'):

        # 验证数据集的预测结果
        self.yanzhneg_pr = []

        # 验证数据集的真实结果
        self.yanzhneg_real = []

        # 预测数据集的预测结果
        self.predi = []

        # 预测数据集的真实结果
        self.preal = []

        # 目标字段名称
        self.zi = zidan

        # 数据结构和数据处理类的保持一致，要把验证数据集的输入和真实的输出合二为一
        self.datai = {}

        # 记录每个模型最终误差的字典
        self.error_dict = OrderedDict()

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

    # 定义均方误差的函数
    def RMSE(self, data1, data2):
        data1, data2 = np.array(data1), np.array(data2)
        subdata = np.power(data1 - data2, 2)
        return np.sqrt(np.sum(subdata) / len(subdata - 1))

    # 随机森林
    def RF_First(self, data, n_estimators=4000, max_features='auto'):
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
            xx = self.RMSE(xul, data[zhe]['train'][:, -1])

            yy = self.RMSE(yanre, data[zhe]['test'][:, -1])

            pp = self.RMSE(prer, data['predict'][:, -1])

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
    def Adaboost_First(self, data, max_depth=50, n_estimators=1000):
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
            model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=max_depth),
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
            xx = self.RMSE(xul, data[zhe]['train'][:, -1])

            yy = self.RMSE(yanre, data[zhe]['test'][:, -1])

            pp = self.RMSE(prer, data['predict'][:, -1])

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
    def GBDT_First(self, data, max_depth=17, n_estimators=57):
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
            model = GradientBoostingRegressor(loss='ls', n_estimators=n_estimators, max_depth=max_depth,
                                              learning_rate=0.12, subsample=0.8)
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
            xx = self.RMSE(xul, data[zhe]['train'][:, -1])

            yy = self.RMSE(yanre, data[zhe]['test'][:, -1])

            pp = self.RMSE(prer, data['predict'][:, -1])

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
    def LightGBM_First(self, data, max_depth=9, n_estimators=380):
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
            model = lgbm.LGBMRegressor(boosting_type='gbdt', objective='regression', num_leaves=1200,
                                       learning_rate=0.17, n_estimators=n_estimators, max_depth=max_depth,
                                       metric='rmse', bagging_fraction=0.8, feature_fraction=0.8, reg_lambda=0.9)

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
            xx = self.RMSE(xul, data[zhe]['train'][:, -1])
            yy = self.RMSE(yanre, data[zhe]['test'][:, -1])
            pp = self.RMSE(prer, data['predict'][:, -1])
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
    def XGBoost_First(self, data, max_depth=50, n_estimators=220):
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
            model = xgb.XGBRegressor(max_depth=max_depth, learning_rate=0.1, n_estimators=n_estimators,
                                     silent=True, objective='reg:gamma')
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
            xx = self.RMSE(xul, data[zhe]['train'][:, -1])
            yy = self.RMSE(yanre, data[zhe]['test'][:, -1])
            pp = self.RMSE(prer, data['predict'][:, -1])
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
    def CatBoost_First(self, data, catsign, depth=8, iterations=80000):

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
            model = cb.CatBoostRegressor(iterations=iterations, depth=depth, learning_rate=0.8, loss_function='RMSE')

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
            xx = self.RMSE(xul, data[zhe]['train'][:, -1])
            yy = self.RMSE(yanre, data[zhe]['test'][:, -1])
            pp = self.RMSE(prer, data['predict'][:, -1])
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

    def __init__(self, in_tr_data, out_tr_data, in_pre_data, out_pre, fenmu, cha):
        self.xdata = in_tr_data
        self.ydata = out_tr_data

        self.xdatapre = in_pre_data
        self.ydapre = out_pre

        self.fen = fenmu

        self.cha = cha
        pass

    # BP神经网络回归
    def BP(self, hiddenlayers=3, hiddennodes=100, learn_rate=0.05, itertimes=50000,
           batch_size=200, activate_func='sigmoid', break_error=0.00000043):
        loss_trrr, loss_pree, sign, fir = bp.Ten_train(self.xdata, self.ydata, self.xdatapre, self.ydapre,
                                                       hiddenlayers=hiddenlayers, hiddennodes=hiddennodes,
                                                       learn_rate=learn_rate, itertimes=itertimes,
                                                       batch_size=batch_size, activate_func=activate_func,
                                                       break_error=break_error)
        # 因为上面的得出的RMSE是数据变换后的，因此要转换到原始的维度
        loss_trrr = np.array(loss_trrr) * self.fen

        loss_pree = np.array(loss_pree) * self.fen

        return loss_trrr, loss_pree, sign, fir

'''
第四部分：绘制图，绘制第一层各个模型中训练，验证数据的误差，
以及最终的预测数据的真实值和误差值的对比
'''
# 定义绘制第一层模型训练、验证、预测数据的误差的函数
# 根据字典绘制不同参数下评分的对比柱状图
def Plot_RMSE_ONE_Stacking(exdict, kaudu=0.2):
    '''
    :param exdict: 不同模型的RMSE 最小二乘回归误差的平方根
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
    ax.set_ylabel('RMSE')
    ax.set_xlabel('Stacking第一层中的模型')
    # 设置刻度
    ax.set_xticks(ind)
    ax.set_xticklabels(palist)

    ax.grid()

    leg = ax.legend(loc='best', ncol=3, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.8)
    plt.title('Stacking第一层中模型的RMSE')
    plt.savefig(r'C:\Users\GWT9\Desktop\Stacking_pm25.jpg')
    bot.file_helper.send_image(r'C:\Users\GWT9\Desktop\Stacking_pm25.jpg')
    return '一层不同模型对比'

# 最终的预测数据的真实值和误差值的对比
# 按照误差值从小到大排列的数据
def Pailie(realr, modelout, count=90):
    '''
    :param real: 预测数据集真实的数据
    :param modelout: 预测数据集的模型的输出值
    :param count: 进行数据对比的条数
    :return: 按照差值从小到大排列的数据
    '''
    relal_num = np.array(realr)
    modelout_num = np.array(modelout)
    # 随机选取
    fu = np.random.choice(list(range(len(realr))), count, replace=False)
    show_real, show_model = relal_num[fu], modelout_num[fu]
    # 计算差值
    sunnum = show_real - show_model
    # 首先组合三个数据列表为字典
    zuhedict = {ii: [show_real[ii], show_model[ii], sunnum[ii]] for ii in range(len(show_model))}
    # 字典按着值排序
    zhenshi = []
    yucede = []
    chazhi = []
    # 按着差值从大到小
    for jj in sorted(zuhedict.items(), key=lambda gy: gy[1][2]):
        zhenshi.append(jj[1][0])
        yucede.append(jj[1][1])
        chazhi.append(jj[1][2])
    return zhenshi, yucede, chazhi

# 绘制预测值，真实值对比折线，以及与两值的误差柱状图
def recspre(yzhenshide, yyucede):
    #  获得展示的数据
    yreal, ypre, cha = Pailie(yzhenshide, yyucede)
    plt.figure()
    ax = plt.subplot(111)
    plt.grid()
    dign = np.arange(len(yreal))
    #  绘制真实值
    ax.scatter(dign, yreal, label='真实值', lw=2, color='blue', marker='*')
    #  绘制预测值
    ax.plot(dign, ypre, label='预测值', lw=2, color='red', linestyle='--', marker='.')
    #  绘制误差柱状图
    ax.bar(dign, cha, 0.1, label='真实值减去预测值', color='k')
    # 绘制0线
    ax.plot(dign, [0] * len(dign), lw=2, color='k')

    ax.set_ylim((int(min(cha)) - 1, int(max([max(yreal), max(ypre)]))))
    ax.set_xlim((0, len(dign)))

    ax.legend(loc='best')
    ax.set_title('北京市Pm2.5预测数据集结果对比')
    plt.savefig(r'C:\Users\GWT9\Desktop\Stacking_duibi.jpg')
    bot.file_helper.send_image(r'C:\Users\GWT9\Desktop\Stacking_duibi.jpg')
    return '完毕'

# 绘制2条对比曲线
def plotcurve(trainone, pretwo):
    fig, axs = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0.1)
    axs[0].plot(range(len(trainone)), trainone, label='训练', lw=3, color='maroon')
    axs[0].legend()
    axs[1].plot(range(len(pretwo)), pretwo, label='验证', lw=3, color='sienna')
    axs[1].legend()
    plt.xlabel('迭代次数')
    axs[0].set_title('第2层模型：BPNN中训练和验证数据的成本函数值')
    plt.savefig(r'C:\Users\GWT9\Desktop\Stacking_errorr.jpg')
    bot.file_helper.send_image(r'C:\Users\GWT9\Desktop\Stacking_errorr.jpg')

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
    data_cat = cat_data.Kfold()  # 折数

    # 开始建立Stacking第一层的模型
    one_stacking = MODELONE()
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
    erce_data.Normal()
    #  将训练数据集分为训练和验证数据集
    redata = erce_data.Kfold()
    #

    # 第二层建模,在这里不在进行交叉验证，因此只选一个数据集
    stacking_two = MODETWO(redata[0]['train'][:, :-1],
                           np.array([redata[0]['train'][:, -1]]).T,
                           redata[0]['test'][:, :-1],
                           np.array([redata[0]['test'][:, -1]]).T,
                           erce_data.fenmu, erce_data.cha)

    # 训练的输出值，预测的输出值, 每一次迭代训练和预测的误差
    lossrain, losspre, signi, gir = stacking_two.BP()



    # 训练完成后读取最优的参数，在计算最终的预测结果
    graph = tf.train.import_meta_graph("./pm25-%s.meta" % signi)
    ses = tf.Session()
    graph.restore(ses, tf.train.latest_checkpoint('./'))
    op_to_restore = tf.get_default_graph().get_tensor_by_name("Sigmoid_%s:0" % gir)  # 这个tensor的名称和激活函数有关系，需要去BP的程序中获得
    w1 = tf.get_default_graph().get_tensor_by_name("x_data:0")
    feed_dict = {w1: redata['predict'][:, :-1]}
    dgsio = ses.run(op_to_restore, feed_dict)

    preout = erce_data.fenmu * dgsio.T[0] + erce_data.cha

    # 绘制第一层中各个模型的误差图
    Plot_RMSE_ONE_Stacking(one_stacking.error_dict)
    # 绘制预测值，真实值对比折线，以及与两值的误差柱状图
    recspre(erce_data.fenmu * erce_data.nodata.values[:, -1] + erce_data.cha, preout)
    # 绘制第二层模型中的训练和验证误差
    plotcurve(lossrain, losspre)