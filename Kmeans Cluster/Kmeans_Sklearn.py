#-*- coding：utf-8 -*-
# &Author  AnFany

#获得数据
from Wine_Data import DATA
import  numpy as np
from sklearn.cluster import KMeans


# 需要将算法输出的类别转换为真实的类别

# 首先得出原始数据中的类别对应的编号
def get_start(ydata):
    in_class = {}
    classtype = sorted(list(set(list(ydata))))
    for du in range(len(classtype)):
        in_class[du+1] = np.arange(len(ydata))[ydata == classtype[du]]
    return in_class

# 因为算法生成的类别和原始的类别的对应关系不知，下面按照最大的重复比来一一确认
def judge(starclass, endclass, ydata):
    newclass = {} #存储判断出类别后的数据
    clasdict = {} # 存储算法生成的类别和真实类别的对应关系的字典
    for ekey in endclass:
        judg = []
        for skey in starclass:
            # 判断和原始类别中的哪一个元素重复比最高
            repeat = [len([val for val in endclass[ekey] if val in starclass[skey]]), skey]
            judg.append(repeat)
        # 选择最大的数，确定类别
        judg = np.array(judg)
        du = judg[judg.argmax(axis=0)[0]][1]  #判断出来属于哪一类
        clasdict[ekey] = du # 算法生成的类别：原始的类别
        newclass[du] = endclass[ekey]
    # 按样本的序号输出其对应的类别
    newdata = np.ones(len(ydata))
    for fgh in newclass:
        for hu in newclass[fgh]:
            newdata[hu] = fgh
    return newdata, clasdict

# 计算混淆矩阵
#计算混淆矩阵
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

# 将sklearn输出的结果变为字典形式
def trans(resu):
    redict = {}
    for ire in range(len(resu)):
        try:
            redict[resu[ire]].append(ire)
        except KeyError:
            redict[resu[ire]] = [ire]
    return redict


sk = KMeans(init='k-means++', n_clusters=3, n_init=10)

train = sk.fit(DATA[0])
result = sk.predict(DATA[0])

init_class = get_start(DATA[1])
kresult = trans(result)

newy = judge(init_class, kresult, DATA[1])


#输出混淆矩阵
print('混淆矩阵：\n', confusion(np.array([DATA[1]]).T, np.array([newy[0]]).T))


# 输出类别中心
print (train.cluster_centers_)
