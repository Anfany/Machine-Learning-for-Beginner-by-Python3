#-*- coding：utf-8 -*-
# &Author  AnFany

import pandas as pd
data = pd.read_csv(r'C:\Users\GWT9\Desktop\Heart.csv')


# 数据说明
# Attributes types
# -----------------
#
# Real: 1,4,5,8,10,12
# Ordered:11,
# Binary: 2,6,9
# Nominal:7,3,13

# 数据处理说明
# Real, Ordered ： 标准化
# Nominal  独热编码
# Binary 不做处理

# Variable to be predicted
# ------------------------
# Absence (1) or presence (2) of heart disease
# 0,1编码

# 开始进行数据处理【没有缺失值】
normal = [1, 4, 5, 8, 10, 12, 11]  # 标准化处理
one_hot = [3, 7, 13] # one_hot编码
binary = [14]  # 不做处理

#数据处理
def trans(exdata, nor=normal, oh=one_hot, bin=binary):
    keylist = exdata.keys()
    newexdata = pd.DataFrame()
    for ikey in range(len(keylist)):
        if ikey + 1 in nor:
            newexdata[keylist[ikey]] = (exdata[keylist[ikey]] - exdata[keylist[ikey]].mean()) / exdata[keylist[ikey]].std()
        elif ikey + 1 in bin:
            newexdata[keylist[ikey]] = [1 if inum == 1 else 0 for inum in exdata[keylist[ikey]]]
        elif ikey + 1 in oh:
            newdata = pd.get_dummies(exdata[keylist[ikey]], prefix=keylist[ikey])
            newexdata = pd.concat([newexdata,newdata], axis=1)
    return newexdata

Data = trans(data).values
x_pre_data = Data[:, :-1]
y_data = Data[:, -1].reshape(-1, 1)

#最终的可用于算法的数据
model_data = [x_pre_data, y_data]


#数据结构
#x_pre_data.shape = (样本数，特征数)
#y_data.shape= (样本数，1)




