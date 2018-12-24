# -*- coding：utf-8 -*-
# &Author  AnFany


import pandas as pd
import numpy as np

#  训练数据文件路径
train_path = 'C:/Users/GWT9\Desktop/Adult_Train.csv'

#  预测数据文件路径
pre_path = 'C:/Users/GWT9\Desktop/Adult_Test.csv'

# 读取并且处理确缺失数据
def ReadHandle(filepath, miss='fill'):  # 定义处理数据的函数
    data = pd.read_csv(r'%s' % filepath)
    data = data.replace('?', np.nan)
    #  处理缺失值
    if miss == 'del':  # 删除掉缺失值
        miss_data = data.dropna(how='any')
    else:
        miss_data = data.fillna(method='ffill')
    return miss_data

# 读取的数据
read_trai = ReadHandle(train_path)
read_pr = ReadHandle(pre_path)

# 因为目标字段"Money"中，预测数据较训练数据多了一个点，需要处理
read_pr["Money"] = [ii[:-1] for ii in read_pr["Money"]]


# 用字典储存数据
data_dict = {}
data_dict['train'] = read_trai
data_dict['predict'] = read_pr