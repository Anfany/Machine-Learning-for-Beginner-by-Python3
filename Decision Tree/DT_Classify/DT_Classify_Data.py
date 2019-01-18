# -*- coding：utf-8 -*-
# &Author  AnFany

import pandas as pd
import numpy as np

#  训练数据文件路径
train_path = 'C:/Users/GWT9\Desktop/Adult_Train.csv'

#  测试数据文件路径
test_path = 'C:/Users/GWT9\Desktop/Adult_Test.csv'

# 读取数据的函数
def handle_data(filepath, miss='fill'):  # 定义处理数据的函数
    data = pd.read_csv(r'%s' % filepath)
    data = data.replace('?', np.nan)
    #  处理缺失值
    if miss == 'del':  # 删除掉缺失值
        miss_data = data.dropna(how='any')
    else:
        miss_data = data.fillna(method='ffill')
    # 因为测试数据和训练数据的标识有些许不同，因此在这里统一
    miss_data['Money'] = ['不高于50K' if '<=' in hh else '高于50K' for hh in miss_data['Money']]
    return miss_data


# 训练数据
train_data = handle_data(train_path).values




# k折交叉，形成字典
def kfold(trdata, percent_test=0.2): # k最小值为5
    kfoldict = {}
    length = len(trdata)
    sign = int(length * percent_test)
    # 生成随机数组
    random_list = np.arange(length)
    np.random.shuffle(random_list)
    kfoldict[0] = {}
    kfoldict[0]['train'] = trdata[random_list[sign:]]
    kfoldict[0]['test'] = trdata[random_list[:sign]]

    return kfoldict

# 第一代数据集测试数据和验证数据

dt_data = kfold(train_data)

# 预测数据
test_data = handle_data(test_path).values

