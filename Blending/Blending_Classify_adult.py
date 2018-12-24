# -*- coding：utf-8 -*-
# &Author  AnFany

# 两层的Blending分类

#  第一层6个模型：随机森林，AdaBoost，GBDT，LightGBM，XGBoost，CatBoost
#  第二层模型：逻辑回归

# 引入数据文件
import adult_Blending_Data as adult

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
# 逻辑回归
import LR

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
        self.k = 0.3  # 和Stacking不同，验证数据集是固定的，在这里设置验证数据集的比例

        # 训练数据
        self.chidata = self.data['train']

        # 预测数据
        self.nodata = self.data['predict']

        # 类别型数据的编号
        self.catsign = self.Sign()

        # 目标字段
        self.ziduan = mubiao

        # 对于CatBoost而言，因为需要将训练数据的字段，转为数字，因此结果中需要将这个数字转化为真实的类别名称
        self.typedict = self.TransType()


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

    # 定义将训练数据集按照比例分为训练、验证数据集的函数，返回不同用途的数据字典
    def Kfold(self):
        # 因为每个模型需要将验证数据结合的结果集成起来，为了方便起见，在这里固定每一折数中的数据集合
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
    def RF_First(self, data, n_estimators=800, max_features='sqrt'):
        # 对训练数据进行训练，返回模验证数据，预测数据的预测结果
        model = RF(n_estimators=n_estimators, max_features=max_features)
        model.fit(data['train'][:, :-1], data['train'][:, -1])
        # 存储验证数据集结果和预测数据集结果
        # 训练数据集的预测结果
        xul = model.predict(data['train'][:, :-1])
        # 验证的预测结果
        yanre = model.predict(data['test'][:, :-1])
        # 预测的预测结果
        prer = model.predict(data['predict'][:, :-1])

        # 每计算一折后，要计算训练、验证、预测数据的误差
        xx = self.F1(xul, data['train'][:, -1])

        yy = self.F1(yanre, data['test'][:, -1])

        pp = self.F1(prer, data['predict'][:, -1])

        # 开始结合
        self.yanzhneg_pr.append(yanre)
        self.yanzhneg_real = data['test'][:, -1]
        self.predi.append(prer)
        self.preal = data['predict'][:, -1]

        # 存储误差
        self.error_dict['随机森林'] = [xx, yy, pp]
        return print('1层中的随机森林运行完毕')

    # AdaBoost
    def Adaboost_First(self, data, max_depth=5, n_estimators=300):
        model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth),
                                   algorithm="SAMME",
                                   n_estimators=n_estimators, learning_rate=0.8)
        model.fit(data['train'][:, :-1], data['train'][:, -1])
        # 存储验证数据集结果和预测数据集结果的
        # 训练数据集的预测结果
        xul = model.predict(data['train'][:, :-1])
        # 验证的预测结果
        yanre = model.predict(data['test'][:, :-1])
        # 预测的预测结果
        prer = model.predict(data['predict'][:, :-1])

        # 每计算一折后，要计算训练、验证、预测数据的误差
        xx = self.F1(xul, data['train'][:, -1])

        yy = self.F1(yanre, data['test'][:, -1])

        pp = self.F1(prer, data['predict'][:, -1])

        # 开始结合
        self.yanzhneg_pr.append(yanre)
        self.yanzhneg_real = data['test'][:, -1]
        self.predi.append(prer)
        self.preal = data['predict'][:, -1]

        # 存储误差
        self.error_dict['AdaBoost'] = [xx, yy, pp]

        return print('1层中的AdaBoost运行完毕')

    # GBDT
    def GBDT_First(self, data, max_depth=5, n_estimators=320):
        model = GradientBoostingClassifier(loss='deviance', n_estimators=n_estimators, max_depth=max_depth,
                                           learning_rate=0.1, max_features='sqrt')
        model.fit(data['train'][:, :-1], data['train'][:, -1])
        # 存储验证数据集结果和预测数据集结果
        # 训练数据集的预测结果
        xul = model.predict(data['train'][:, :-1])
        # 验证的预测结果
        yanre = model.predict(data['test'][:, :-1])
        # 预测的预测结果
        prer = model.predict(data['predict'][:, :-1])

        # 每计算一折后，要计算训练、验证、预测数据的误差
        xx = self.F1(xul, data['train'][:, -1])

        yy = self.F1(yanre, data['test'][:, -1])

        pp = self.F1(prer, data['predict'][:, -1])

        # 开始结合
        self.yanzhneg_pr.append(yanre)
        self.yanzhneg_real = data['test'][:, -1]
        self.predi.append(prer)
        self.preal = data['predict'][:, -1]

        # 存储误差
        self.error_dict['GBDT'] = [xx, yy, pp]


        return print('1层中的GBDT运行完毕')

    # LightGBM
    def LightGBM_First(self, data, max_depth=5, n_estimators=400):
        model = lgbm.LGBMClassifier(boosting_type='gbdt', objective='binary', num_leaves=200,
                                    learning_rate=0.1, n_estimators=n_estimators, max_depth=max_depth,
                                    bagging_fraction=0.9, feature_fraction=0.9, reg_lambda=0.2)
        model.fit(data['train'][:, :-1], data['train'][:, -1])
        # 存储验证数据集结果和预测数据集结果
        # 训练数据集的预测结果
        xul = model.predict(data['train'][:, :-1])
        # 验证的预测结果
        yanre = model.predict(data['test'][:, :-1])
        # 预测的预测结果
        prer = model.predict(data['predict'][:, :-1])

        # 每计算一折后，要计算训练、验证、预测数据的误差
        xx = self.F1(xul, data['train'][:, -1])

        yy = self.F1(yanre, data['test'][:, -1])

        pp = self.F1(prer, data['predict'][:, -1])

        # 开始结合
        self.yanzhneg_pr.append(yanre)
        self.yanzhneg_real = data['test'][:, -1]
        self.predi.append(prer)
        self.preal = data['predict'][:, -1]

        # 存储误差
        self.error_dict['LightGBM'] = [xx, yy, pp]

        return print('1层中的LightGBM运行完毕')

    # XGBoost
    def XGBoost_First(self, data, max_depth=8, n_estimators=220):
        model = xgb.XGBClassifier(max_depth=max_depth, learning_rate=0.1, n_estimators=n_estimators,
                                  silent=True, objective='binary:logistic', booster='gbtree')
        model.fit(data['train'][:, :-1], data['train'][:, -1])
        # 存储验证数据集结果和预测数据集结果
        # 训练数据集的预测结果
        xul = model.predict(data['train'][:, :-1])
        # 验证的预测结果
        yanre = model.predict(data['test'][:, :-1])
        # 预测的预测结果
        prer = model.predict(data['predict'][:, :-1])

        # 每计算一折后，要计算训练、验证、预测数据的误差
        xx = self.F1(xul, data['train'][:, -1])

        yy = self.F1(yanre, data['test'][:, -1])

        pp = self.F1(prer, data['predict'][:, -1])

        # 开始结合
        self.yanzhneg_pr.append(yanre)
        self.yanzhneg_real = data['test'][:, -1]
        self.predi.append(prer)
        self.preal = data['predict'][:, -1]

        # 存储误差
        self.error_dict['XGBoost'] = [xx, yy, pp]
        return print('1层中的XGBoost运行完毕')

    # CatBoost
    def CatBoost_First(self, data, catsign, depth=5, iterations=200):

        model = cb.CatBoostClassifier(iterations=iterations, depth=depth, learning_rate=0.5,
                                      loss_function='Logloss', logging_level='Verbose')

        model.fit(data['train'][:, :-1], data['train'][:, -1], cat_features=catsign)
        # 注意存储验证数据集结果和预测数据集结果的不同
        # 训练数据集的预测结果
        xul = model.predict(data['train'][:, :-1])
        # 验证的预测结果
        yanre = model.predict(data['test'][:, :-1])
        # 预测的预测结果
        prer = model.predict(data['predict'][:, :-1])

        # 每计算一折后，要计算训练、验证、预测数据的误差
        xx = self.F1(xul, data['train'][:, -1])

        yy = self.F1(yanre, data['test'][:, -1])

        pp = self.F1(prer, data['predict'][:, -1])

        # 开始结合
        self.yanzhneg_pr.append(yanre)
        self.yanzhneg_real = data['test'][:, -1]
        self.predi.append(prer)
        self.preal = data['predict'][:, -1]

        # 存储误差
        self.error_dict['CatBoost'] = [xx, yy, pp]

        return print('1层中的CatBoost运行完毕')

'''
第三部分：第二层的模型运行阶段 可以任意更换模型
'''
class MODETWO:

    def __init__(self, in_tr_data, out_tr_data, in_pre_data):
        self.xdata = in_tr_data
        self.ydata = out_tr_data

        self.xdatapre = in_pre_data


    # 定义逻辑回归的函数
    def Lr(self, learn_rate=0.5, iter_tiems=40000, error=1e-9, con='L2'):
        losserr, preout = LR.trans_tf(self.xdata, self.ydata, self.xdatapre,
                                      learn_rate=learn_rate, iter_tiems=iter_tiems,
                                      error=error, con=con)
        return losserr, preout


'''
第四部分：绘制图，绘制第一层各个模型中训练，验证数据的误差，
以及最终的预测数据的真实值和误差值的对比
'''
# 定义绘制第一层模型训练、验证、预测数据的F1度量的函数
# 根据字典绘制不同参数下评分的对比柱状图
def Plot_RMSE_ONE_Blending(exdict, kaudu=0.2):
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
    ax.set_ylabel('f1度量')
    ax.set_xlabel('Blending第一层中的模型')
    # 设置刻度
    ax.set_xticks(ind)
    ax.set_xticklabels(palist)

    leg = ax.legend(loc='best', ncol=3, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.8)

    plt.title('Blending第一层中模型的f1度量')
    plt.savefig(r'C:\Users\GWT9\Desktop\Blending_adult.jpg')
    return print('一层不同模型对比')

# 绘制预测值，真实值对比折线，以及与两值的误差柱状图
def recspre(yzhenshide):
    plt.figure()
    #  绘制真实值
    plt.plot(list(range(len(yzhenshide))), yzhenshide, lw=2, color='blue', marker='*')
    plt.title('逻辑回归成本函数')
    plt.savefig(r'C:\Users\GWT9\Desktop\Blending_duibi.jpg')
    return '完毕'
'''
第五部分：Blending主函数
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


    # 开始建立Blending第一层的模型
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
    erce_data.k = 0  # # 不再设置验证数据集
    bpdatadict = erce_data.Kfold()

    # 第二层建模,
    stacking_two = MODETWO(bpdatadict['train'][:, :-1],
                           np.array([bpdatadict['train'][:, -1]]).T,
                           bpdatadict['predict'][:, :-1])

    # 训练的输出值，预测的输出值, 每一次迭代训练和预测的误差
    error_acc, preoutput = stacking_two.Lr()

    #  将输出的结果转变为真实的类别，输出混淆矩阵
    bp_out_type = one_stacking.AntiTae(np.array(preoutput).T[0])
    bp_real_type = one_stacking.AntiTae(bpdatadict['predict'][:, -1])

    # 绘制第一层中各个模型的误差图
    Plot_RMSE_ONE_Blending(one_stacking.error_dict)
    # 绘制第二层模型中的训练过程中的成本函数的下降
    recspre(error_acc)
    fru = one_stacking.ConfuseMatrix(bp_real_type, bp_out_type)
    print(fru)
    print(one_stacking.F1(bp_real_type, bp_out_type))



