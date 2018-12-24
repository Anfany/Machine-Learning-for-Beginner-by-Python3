# -*- coding：utf-8 -*-
# &Author  AnFany

# 两层的Blending回归

#  第一层7个模型：随机森林，AdaBoost，GBDT，LightGBM，XGBoost，CatBoost，BPNN回归
#  第二层模型：线性回归

# 引入数据文件
import pm25_Blending_data as pm25

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

# 线性回归
import Linear_Regression as linear

# 其他库包
import tensorflow as tf
import numpy as np
import pandas as pd
from collections import OrderedDict  # python字典是无序的，此包是有序的
import os
os.chdir(r'E:\tensorflow_Learn\Blending\pm25')
'''
第一部分：数据处理模型
'''

class DATA:

    def __init__(self, datadict=pm25.data_dict, mubiao='pm2.5'):
        self.data = datadict
        self.k = 0.3  # 在Blending中验证数据所占的比例

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

    # 按照验证数据集的比例,将训练数据集分为验证和训练数据集
    def Kfold(self):
        # 因为要保证每个模型的验证数据是一样的，
        datanum = self.chidata.values
        # 数据集总长度
        length = len(datanum)
        alist = np.arange(length)
        np.random.seed(1990)
        np.random.shuffle(alist)  # 随机打乱数据对BPNN,SVM而言是有益处的，而对于决策树之类的模型而言没有影响

        # 验证数据的长度
        yanlem = int(length * self.k)

        # 存储数据集的字典
        datai = {}
        datai['predict'] = self.nodata.values

        # 选中的作为验证数据集的索引序列
        np.random.seed(2000)
        yanlis = np.random.choice(alist, yanlem, replace=False)

        # 没有被选中的作为训练数据集的索引序列
        trainlis = [ki for ki in alist if ki not in yanlis]

        # 储存训练数据集
        datai['train'] = datanum[trainlis]

        # 储存验证数据集
        datai['test'] = datanum[yanlis]

        # 返回的数据集形式{'train':data, 'test':data, 'predict':data}
        print('数据集分割处理完毕')
        return datai


'''
第二部分：第一层的模型运行阶段
'''
# 可以任意添加模型
class MODELONE:

    def __init__(self, fenmu, cha, zidan='pm2.5'):

        # 验证数据集的预测结果
        self.yanzhneg_pr = []

        # 预测数据集的预测结果
        self.predi = []

        # 目标字段名称
        self.zi = zidan

        # 数据结构和数据处理类的保持一致，要把验证数据集的输入和真实的输出合二为一
        self.datai = {}

        # 记录每个模型最终误差的字典
        self.error_dict = OrderedDict()

        # 验证数据集的真实输出结果
        self.yanzhneg_real = []

        # 预测数据集的真实输出结果
        self.preal = []

        #  针对BPnn，得到的结果数据要反归一化
        self.fen = fenmu
        self.cha = cha

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
    def RF_First(self, data, n_estimators=400, max_features='auto'):
        # 对训练数据进行训练，返回模验证数据，预测数据的预测结果
        model = RF(n_estimators=n_estimators, max_features=max_features)
        model.fit(data['train'][:, :-1], data['train'][:, -1])
        # 注意存储验证数据集结果和预测数据集结果的不同
        # 训练数据集的预测结果
        xul = model.predict(data['train'][:, :-1])
        # 验证的预测结果
        yanre = model.predict(data['test'][:, :-1])
        #预测的预测结果
        prer = model.predict(data['predict'][:, :-1])
        # 储存
        self.yanzhneg_pr.append(yanre)
        self.predi.append(prer)
        # 分别计算训练、验证、预测的误差
        # 每计算一折后，要计算训练、验证、预测数据的误差
        xx = self.RMSE(xul, data['train'][:, -1])
        yy = self.RMSE(yanre, data['test'][:, -1])
        pp = self.RMSE(prer, data['predict'][:, -1])
        # 储存误差
        self.error_dict['随机森林'] = [xx, yy, pp]

        # 验证数据集的真实输出结果
        self.yanzhneg_real = data['test'][:, -1]

        # 预测数据集的真实输出结果
        self.preal = data['predict'][:, -1]

        return print('1层中的随机森林运行完毕')

    # AdaBoost
    def Adaboost_First(self, data, max_depth=5, n_estimators=320):
        model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=max_depth),
                                  n_estimators=n_estimators, learning_rate=0.8)
        model.fit(data['train'][:, :-1], data['train'][:, -1])
        # 注意存储验证数据集结果和预测数据集结果的不同
        # 训练数据集的预测结果
        xul = model.predict(data['train'][:, :-1])
        # 验证的预测结果
        yanre = model.predict(data['test'][:, :-1])
        # 预测的预测结果
        prer = model.predict(data['predict'][:, :-1])
        # 储存
        self.yanzhneg_pr.append(yanre)
        self.predi.append(prer)
        # 分别计算训练、验证、预测的误差
        # 每计算一折后，要计算训练、验证、预测数据的误差
        xx = self.RMSE(xul, data['train'][:, -1])
        yy = self.RMSE(yanre, data['test'][:, -1])
        pp = self.RMSE(prer, data['predict'][:, -1])
        # 储存误差
        self.error_dict['AdaBoost'] = [xx, yy, pp]
        # 验证数据集的真实输出结果
        self.yanzhneg_real = data['test'][:, -1]

        # 预测数据集的真实输出结果
        self.preal = data['predict'][:, -1]
        return print('1层中的AdaBoost运行完毕')

    # GBDT
    def GBDT_First(self, data, max_depth=17, n_estimators=57):
        model = GradientBoostingRegressor(loss='ls', n_estimators=n_estimators, max_depth=max_depth,
                                          learning_rate=0.12, subsample=0.8)
        model.fit(data['train'][:, :-1], data['train'][:, -1])
        # 注意存储验证数据集结果和预测数据集结果的不同
        # 训练数据集的预测结果
        xul = model.predict(data['train'][:, :-1])
        # 验证的预测结果
        yanre = model.predict(data['test'][:, :-1])
        # 预测的预测结果
        prer = model.predict(data['predict'][:, :-1])
        # 储存
        self.yanzhneg_pr.append(yanre)
        self.predi.append(prer)
        # 分别计算训练、验证、预测的误差
        # 每计算一折后，要计算训练、验证、预测数据的误差
        xx = self.RMSE(xul, data['train'][:, -1])
        yy = self.RMSE(yanre, data['test'][:, -1])
        pp = self.RMSE(prer, data['predict'][:, -1])
        # 储存误差
        self.error_dict['GBDT'] = [xx, yy, pp]
        # 验证数据集的真实输出结果
        self.yanzhneg_real = data['test'][:, -1]

        # 预测数据集的真实输出结果
        self.preal = data['predict'][:, -1]
        return print('1层中的GBDT运行完毕')

    # LightGBM
    def LightGBM_First(self, data, max_depth=9, n_estimators=380):
        model = lgbm.LGBMRegressor(boosting_type='gbdt', objective='regression', num_leaves=1200,
                                   learning_rate=0.17, n_estimators=n_estimators, max_depth=max_depth,
                                   metric='rmse', bagging_fraction=0.8, feature_fraction=0.8, reg_lambda=0.9)
        model.fit(data['train'][:, :-1], data['train'][:, -1])
        # 注意存储验证数据集结果和预测数据集结果的不同
        # 训练数据集的预测结果
        xul = model.predict(data['train'][:, :-1])
        # 验证的预测结果
        yanre = model.predict(data['test'][:, :-1])
        # 预测的预测结果
        prer = model.predict(data['predict'][:, :-1])
        # 储存
        self.yanzhneg_pr.append(yanre)
        self.predi.append(prer)
        # 分别计算训练、验证、预测的误差
        # 每计算一折后，要计算训练、验证、预测数据的误差
        xx = self.RMSE(xul, data['train'][:, -1])
        yy = self.RMSE(yanre, data['test'][:, -1])
        pp = self.RMSE(prer, data['predict'][:, -1])
        # 储存误差
        self.error_dict['LightGBM'] = [xx, yy, pp]
        # 验证数据集的真实输出结果
        self.yanzhneg_real = data['test'][:, -1]

        # 预测数据集的真实输出结果
        self.preal = data['predict'][:, -1]
        return print('1层中的LightGBM运行完毕')

    # XGBoost
    def XGBoost_First(self, data, max_depth=5, n_estimators=320):
        model = xgb.XGBRegressor(max_depth=max_depth, learning_rate=0.1, n_estimators=n_estimators,
                                 silent=True, objective='reg:gamma')
        model.fit(data['train'][:, :-1], data['train'][:, -1])
        # 注意存储验证数据集结果和预测数据集结果的不同
        # 训练数据集的预测结果
        xul = model.predict(data['train'][:, :-1])
        # 验证的预测结果
        yanre = model.predict(data['test'][:, :-1])
        # 预测的预测结果
        prer = model.predict(data['predict'][:, :-1])
        # 储存
        self.yanzhneg_pr.append(yanre)
        self.predi.append(prer)
        # 分别计算训练、验证、预测的误差
        # 每计算一折后，要计算训练、验证、预测数据的误差
        xx = self.RMSE(xul, data['train'][:, -1])
        yy = self.RMSE(yanre, data['test'][:, -1])
        pp = self.RMSE(prer, data['predict'][:, -1])
        # 储存误差
        self.error_dict['XGBoost'] = [xx, yy, pp]
        # 验证数据集的真实输出结果
        self.yanzhneg_real = data['test'][:, -1]

        # 预测数据集的真实输出结果
        self.preal = data['predict'][:, -1]
        return print('1层中的XGBoost运行完毕')

    # CatBoost
    def CatBoost_First(self, data, catsign, depth=8, iterations=80000):
        model = cb.CatBoostRegressor(iterations=iterations, depth=depth, learning_rate=0.8, loss_function='RMSE')
        model.fit(data['train'][:, :-1], data['train'][:, -1], cat_features=catsign)
        # 注意存储验证数据集结果和预测数据集结果的不同
        # 训练数据集的预测结果
        xul = model.predict(data['train'][:, :-1])
        # 验证的预测结果
        yanre = model.predict(data['test'][:, :-1])
        # 预测的预测结果
        prer = model.predict(data['predict'][:, :-1])
        # 储存
        self.yanzhneg_pr.append(yanre)
        self.predi.append(prer)
        # 分别计算训练、验证、预测的误差
        # 每计算一折后，要计算训练、验证、预测数据的误差
        xx = self.RMSE(xul, data['train'][:, -1])
        yy = self.RMSE(yanre, data['test'][:, -1])
        pp = self.RMSE(prer, data['predict'][:, -1])
        # 储存误差
        self.error_dict['CatBoost'] = [xx, yy, pp]
        # 验证数据集的真实输出结果
        self.yanzhneg_real = data['test'][:, -1]

        # 预测数据集的真实输出结果
        self.preal = data['predict'][:, -1]
        return print('1层中的CatBoost运行完毕')


    # BPNN回归
    def BPnn(self, data, hiddenlayers=3, hiddennodes=100, learn_rate=0.05, itertimes=10000,
             batch_size=200, activate_func='sigmoid', break_error=0.00000043):
        sign, fir, trer, ader = bp.Ten_train(data['train'][:, :-1], np.array([data['train'][:, -1]]).T,
                                 data['test'][:, :-1], np.array([data['test'][:, -1]]).T,
                                 hiddenlayers=hiddenlayers, hiddennodes=hiddennodes,
                                 learn_rate=learn_rate, itertimes=itertimes,
                                 batch_size=batch_size, activate_func=activate_func,
                                 break_error=break_error)

        # 这个模型的误差在后文添加上
        return sign, fir, trer, ader

'''
第三部分：第二层的模型运行阶段 可以任意更换模型
'''
class MODETWO:

    def __init__(self, in_tr_data, out_tr_data, in_pre_data):
        self.xdata = in_tr_data
        self.ydata = out_tr_data

        self.in_pre_data = in_pre_data


    # 梯度下降线性回归
    def Lin_Gr(self, learn_rate=0.00000003, iter_times=80000, error=1e-9):
        model = linear.LinearRegression(learn_rate=learn_rate, iter_times=iter_times, error=error)
        # 两种方法
        model.Gradient(self.xdata, self.ydata)
        outpre = model.predict(self.in_pre_data)
        return outpre

    # 公式法线性回归
    def Lin_Fo(self):
        model = linear.LinearRegression()
        # 两种方法
        model.Formula(self.xdata, self.ydata)
        outpre = model.predict(self.in_pre_data)
        return outpre



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
    ax.set_xlabel('Blending第一层中的模型')
    # 设置刻度
    ax.set_xticks(ind)
    ax.set_xticklabels(palist)

    ax.grid()

    leg = ax.legend(loc='best', ncol=3, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.8)
    plt.title('Blending第一层中模型的RMSE')
    plt.savefig(r'C:\Users\GWT9\Desktop\Blending_pm25.jpg')
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
    np.random.seed(200)
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
def recspre(yzhenshide, yyucede, title='公式法'):
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
    ax.set_title('%s:北京市Pm2.5预测数据集结果对比' % title)
    plt.savefig(r'C:\Users\GWT9\Desktop\Blending_duibi_%s.jpg' % title)

    return '完毕'



'''
第五部分：Stacking主函数
'''

if __name__ == "__main__":
    #  第一层6个模型：随机森林，AdaBoost，GBDT，LightGBM，XGBoost，CatBoost

    # 下面依次为每个模型建立数据
    # 随机森林、AdaBoost，GBDT，LIghtGNM，XGBoost都是一样的
    rf_data = DATA()
    rf_data.CAtoDI()  # 标签数字化
    data_rf = rf_data.Kfold()  # 分割数据

    # CatBoost
    cat_data = DATA()  # 不用处理
    data_cat = cat_data.Kfold()  # 分割数据


    # BPnn数据处理
    bp_data = DATA()
    bp_data.CAtoDI()  # 标签数字化
    bp_data.Stand()   # 标准化
    data_bp = bp_data.Kfold()  # 分割数据

    # 开始建立Stacking第一层的模型
    one_stacking = MODELONE(bp_data.fenmu, bp_data.cha)
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

    # BPnn
    signi, gir, trarm, adder = one_stacking.BPnn(data_bp)
    # 此处要下载最优的BPnn的模型，计算成本函数值，验证、预测数据集的预测结果
    # 训练完成后读取最优的参数，在计算最终的预测结果
    graph = tf.train.import_meta_graph("./pm25-%s.meta" % signi)
    ses = tf.Session()
    graph.restore(ses, tf.train.latest_checkpoint('./'))
    op_to_restore = tf.get_default_graph().get_tensor_by_name("Sigmoid_%s:0" % gir)  # 这个tensor的名称和激活函数有关系，需要去BP的程序中获得
    w1 = tf.get_default_graph().get_tensor_by_name("x_data:0")
    feed_dict = {w1: data_bp['predict'][:, :-1]}
    dgsio = ses.run(op_to_restore, feed_dict)

    preout = bp_data.fenmu * dgsio.T[0] + bp_data.cha

    # 在这里需要计算预测数据集的误差
    prerror = one_stacking.RMSE(preout, data_bp['predict'][:, -1] * bp_data.fenmu + bp_data.cha)

    one_stacking.error_dict['BPNN'] = [trarm * bp_data.fenmu, adder * bp_data.fenmu, prerror]  # 因为BPNN的误差是数据处理后的误差

    # 绘制第一层中各个模型的误差图
    Plot_RMSE_ONE_Stacking(one_stacking.error_dict)

    feed_dict_add = {w1: data_bp['test'][:, :-1]}
    dgsio_add = ses.run(op_to_restore, feed_dict_add)
    adderout = bp_data.fenmu * dgsio_add.T[0] + bp_data.cha

    # 添加BPNN模型的验证、预测数据集的输出结果
    one_stacking.yanzhneg_pr.append(adderout)
    one_stacking.predi.append(preout)


    # 第二层的数据准备
    one_stacking.DataStru()
    data_two = one_stacking.datai

    # 第二层的数据处理
    erce_data = DATA(datadict=data_two)
    #  因为线性回归是可以取到最优值的，因此在里不在设置验证数据集
    erce_data.k = 0
    redata = erce_data.Kfold()

    # 第二层建模
    stacking_two = MODETWO(np.array(redata['train'][:, :-1]),
                           np.array([redata['train'][:, -1]]).T,
                           np.array(redata['predict'][:, :-1]))

    # 两种不同方法的线性回归:公式法
    outlist_Fo = stacking_two.Lin_Fo()

    # 输出最终的对比曲线
    recspre(redata['predict'][:, -1], outlist_Fo.T[0], title='公式法')


    # 两种不同方法的线性回归:梯度下降法
    outlist_Gr = stacking_two.Lin_Gr()


    # 输出最终的对比曲线
    recspre(redata['predict'][:, -1], outlist_Gr.T[0], title='梯度下降法')