#-*- coding：utf-8 -*-
# &Author  AnFany

#获得数据
from Wine_Data import DATA
import numpy as np


# 定义欧几里得距离
def dis(sample, center):
    cen = np.array([center])
    sample = np.array(sample)
    if len(sample) != 0:
        usb = np.sum((sample - cen) ** 2, axis=1) ** 0.5
        return usb
    else:
        return 0

# 定义根据距离列表，概率较大的被选中
def selec(dislist):
    #首先将所有数值除以距离和
    divided = dislist / np.sum(dislist)
    # 随机选取0-1之内的数字
    num = np.random.random()
    for hh in range(len(divided)):
        num -= divided[hh]
        if num < 0:
            return hh

# 定义生成初始的聚类中心的函数
def gencenter(sample, type):
    # 随机选择初始的样本编号
    sign = np.random.choice(list(range(len(sample))), 1)[0]
    #存储类别中心的数组
    centerlist = [sample[sign]]
    while len(centerlist) < type:
        # 添加新的
        distance = dis(sample, centerlist[-1])  # 和刚添加的中心计算距离
        newsign = selec(distance)
        centerlist.append(sample[newsign])
    return np.array(centerlist)

# Kmeans++聚类算法
def kmeans(samp, maxtimes, costerror, countcenter):
    # kmeans++ 产生出的初始的类别中心
    center = gencenter(samp, type=countcenter)

    # 存储成本函数的值
    costfunc = []

    iter = 0

    while iter < maxtimes:
        # 开始根据类别中心匹配距离
        samdict  = {}
        signdict = {}
        # 每个类别 定义成一个集合
        for jj in range(len(center)):
            samdict[jj] = [] # 存储样本
            signdict[jj] = [] # 存储样本编号
        # 为每一个样本计算类别
        dictgn = 0
        for hg in samp:
            ddis = dis(center, hg) #计算样本与每一个类别中心的距离
            # 找到最小的
            minsign = ddis.argmin()
            samdict[minsign].append(hg)  # 添加到该类别的样本集合中
            signdict[minsign].append(dictgn)
            dictgn += 1


        # 计算此时分类结果的cost
        cost = 0
        for cc in samdict:
            cost += np.sum(dis(samdict[cc], center[cc]))

        # 存储cost
        costfunc.append(cost)

        # 判断是否提前结束迭代
        if len(costfunc) > 2:
            if 0 <= costfunc[-2] - costfunc[-1] < costerror:
                break

         # 更新类别中心
        for kk in samdict:
            if len(signdict[kk]) != 0:
                center[kk] = np.mean(samdict[kk], axis=0)  # 均值

        iter += 1

    return center, costfunc, signdict

# 因为Kmeans 算法不保证每一次都取得最优值。因此定义运行的次数，选择cost最小的
def op_kmeans(saple, maxti=1000, costerr=1e-19, countcen=3, maxtimes=90):
    times = 0
    # 存储cost
    costff = [1e9]

    #最优的结果lastre
    lastre = 0
    while times < maxtimes:
        step = kmeans(saple, maxtimes=maxti, costerror=costerr, countcenter=countcen)
        if len(costff) != 0:
            if costff[0] > step[1][-1]:
                lastre = step
                costff = [step[1][-1]]
        else:
            costff = [step[1][-1]]
        times += 1
    return lastre


# 结果验证

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
def confusion(realy, outy, method='AnFany'):
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

#  最终的程序
if __name__ == "__main__":
    init_class = get_start(DATA[1])
    kresult = op_kmeans(DATA[0])
    newy = judge(init_class, kresult[2], DATA[1])

    # #输出混淆矩阵
    print('混淆矩阵：\n', confusion(np.array([DATA[1]]).T, np.array([newy[0]]).T))

    # 输出最后计算得到的真实类别的类别中心
    for real in kresult[2]:
        print('类别%s的中心为：\n%s' % (newy[1][real], kresult[0][real]))

    # 绘制成本函数图
    import matplotlib.pyplot as plt
    from pylab import mpl  # 作图显示中文

    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 设置中文字体新宋体
    mpl.rcParams['axes.unicode_minus'] = False
    plt.plot(list(range(len(kresult[1]))), kresult[1], '-', linewidth=5)
    plt.title('成本函数图')
    plt.ylabel('Cost 值')
    plt.xlabel('迭代次数')
    plt.show()

