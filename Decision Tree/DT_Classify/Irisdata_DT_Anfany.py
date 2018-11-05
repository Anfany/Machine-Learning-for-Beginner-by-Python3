#-*- coding：utf-8 -*-
# &Author  AnFany

import pandas as pd
import numpy as np

data = pd.read_csv(r'C:\Users\GWT9\Desktop\iris.csv')

# 直接将数据按7:2:1 的比例分为训练、验证、测试数据

def fenge(data, per=[0.2, 0.1]):
    # 总长度
    lent = len(data)
    alist = np.arange(lent)
    np.random.shuffle(alist)

    # 验证
    shu = int(lent * per[0])
    yu = int(lent * per[1])

    yanzheng = np.random.choice(alist, shu)

    # 预测
    shengxai = np.array([i for i in alist if i not in yanzheng])

    yuce = np.random.choice(shengxai, yu)

    # 训练
    train = np.array([j for j in alist if j not in yanzheng and j not in yuce])

    # 返回数据集
    dadata = {}
    dadata[0] = {}

    dadata[0]['train'] = data[train]
    dadata[0]['test'] = data[yanzheng]

    return dadata, data[yuce]

deeer = fenge(data.values[:, 1:])

# 预测数据
dt_data = deeer[0]
test_data = deeer[1]