#-*- coding：utf-8 -*-
# &Author  AnFany

import sklearn as sk
from Iris_Data import Data as smdata
import numpy as np

from sklearn.linear_model import LogisticRegression

sklr = LogisticRegression(multi_class='multinomial', solver='sag', C=200, max_iter=10000)

#格式化输出混淆矩阵
from prettytable import PrettyTable
def confusion(realy, outy, method='Sklearn'):
    mix = PrettyTable()
    type = sorted(list(set(realy.T[0])), reverse=True)
    mix.field_names = [method] + ['预测:%d类'%si for si in type]
    # 字典形式存储混淆矩阵数据
    cmdict = {}
    for jkj in type:
        cmdict[jkj] = []
        for hh in type:
            hu = len(['0' for jj in range(len(realy)) if realy[jj][0] == jkj and outy[jj][0] == hh])
            cmdict[jkj].append(hu)
    # 输出表格
    for fu in type:
        mix.add_row(['真实:%d类'%fu] + cmdict[fu])
    return mix


# 将独热编码的类别变为标识为1，2，3的类别
def transign(eydata):
    ysign = []
    for hh in eydata:
        ysign.append([list(hh).index(1) + 1])
    return np.array(ysign)

regre = sklr.fit(smdata[0], transign(smdata[1]).T[0])

predata = np.array([sklr.predict(smdata[0])]).T


print('系数为：\n', np.hstack((sklr.coef_, np.array([sklr.intercept_]).T)).T)


print('混淆矩阵：\n',confusion(transign(smdata[1]), predata))
