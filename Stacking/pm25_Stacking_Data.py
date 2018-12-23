# -*- coding：utf-8 -*-
# &Author  AnFany

import pandas as pd
import numpy as np

# 读取数据文件
data = pd.read_csv(r'C:\Users\GWT9\Desktop\PRSA_data_2010.1.1-2014.12.31.csv')


'''第一部分：缺失值的处理'''
#  因为Pm2.5是目标数据，如有缺失值直接删除这一条记录

# 删除目标值为空值的行, 其他列为缺失值则自动填充,并将目标变量放置在数据集最后一列
def DeleteTargetNan(exdata, targetstr):
    '''
    :param exdata: dataframe数据集
    :param targetstr: 目标字段名称
    :return: 预处理后的dataframe格式的数据集
    '''
    #  首先判断目标字段是否有缺失值
    if exdata[targetstr].isnull().any():
        #  首先确定缺失值的行数
        loc = exdata[targetstr][data[targetstr].isnull().values == True].index.tolist()
        #  然后删除这些行
        exdata = exdata.drop(loc)
    # 凡是有缺失值的再一起利用此行的均值填充
    exdata = exdata.fillna(exdata.mean())
    # 将目标字段至放在最后的一列
    targetnum = exdata[targetstr].copy()
    del exdata[targetstr]
    exdata[targetstr] = targetnum
    return exdata

# 删除原始数据中不需要的字段名
def Shanchu(exdata, aiduan=['No']):
    for ai in aiduan:
        if ai in exdata.keys():
            del exdata[ai]
    return exdata


# 数据处理后最终的数据集
first = DeleteTargetNan(data, 'pm2.5')
# 去除字段后
two = Shanchu(first)

# 因为此数据集没有提供预测数据集，因此在总的数据集合中随机选择一部分作为预测数据集
# 对于Stacking，因为需要保证每一次的预测数据是固定的，因此在这里固定随机种子，保证预测数据是固定的
# 对于训练数据的k折处理以及数据的处理，放在模型的文件中

def fenge(exdata, per=0.2):
    '''
    :param exdata: 总的数据集DataFrame格式
    :param per: 预测数据占的比例
    :return: 返回{'train':dataframe, 'predict':dataframe}样式的字典
    '''

    np.random.seed(1000)
    df_predict = exdata.sample(frac=per)

    rowlist = []
    for indexs in df_predict.index:
        rowlist.append(indexs)
    df_train = exdata.drop(rowlist, axis=0)

    # 保存数据的字典
    datict = {}
    datict['train'] = df_train
    datict['predict'] = df_predict

    return datict

# 数据字典
data_dict = fenge(two)





