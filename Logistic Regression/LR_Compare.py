# -*- coding：utf-8 -*-
# &Author  AnFany

# 引入三种方法
import LR_AnFany as LR_A  # AnFany
import LR_Sklearn as LR_S # Sklearn
import LR_TensorFlow as LR_T # TensorFlow
from pylab import mpl  # 作图显示中文
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 设置中文字体新宋体
mpl.rcParams['axes.unicode_minus'] = False


import numpy as np
#随机生成二元二分类的数据: 训练数据和预测数据的比例为6:4。预测数据不能重新随机生成，因为分布不同了。
#np.random.seed(828)
x_data = np.random.random((900, 2))
y_data = np.array([[1] if 0.3*a[0] + 0.6*a[1] + 0.55 >= 1 else [0] for a in x_data])

#拆分为训练数据集和预测数据集
def divided(xdata, ydata, percent=0.4):
    sign_list = list(range(len(xdata)))
    #用于测试的序号
    select_sign = sorted(np.random.choice(sign_list, int(len(xdata)*percent), replace=False))

    #用于训练的序号
    no_select_sign = [isign for isign in sign_list if isign not in select_sign]

    #测试数据
    x_predict_data = xdata[select_sign]
    y_predict_data = ydata[select_sign].reshape(len(select_sign), 1)#转化数据结构

    #训练数据
    x_train_data = xdata[no_select_sign]
    y_train_data = ydata[no_select_sign].reshape(len(no_select_sign), 1)#转化数据结构

    return x_train_data, y_train_data, x_predict_data, y_predict_data #训练的x，y;  测试的x，y


alldata = divided(x_data, y_data)


#训练的Xdata
train_x_data, train_y_data, pre_x_data, pre_y_data = alldata

#########不同的方法训练

#AnFany
lr_re = LR_A.LRReg()
lf = lr_re.Gradient(train_x_data, train_y_data)
datd = lr_re.predict(pre_x_data)
# print('混淆矩阵：\n', LR_A.confusion(pre_y_data, datd))


#Sklearn
regre = LR_S.sklr.fit(train_x_data, train_y_data.T[0])
pfdata = LR_S.sklr.predict(pre_x_data)
weig = regre.coef_.T
weig = np.array([np.append(weig, regre.intercept_)]).T
dm = LR_S.confusion_matrix(pre_y_data.T[0], pfdata)
# print('混淆矩阵：\n', LR_S.confusion(dm))


#TensorFlow
ypre = LR_T.trans_tf(train_x_data, train_y_data, pre_x_data)
tfweib = np.vstack((ypre[2], ypre[3]))
# print('混淆矩阵：\n', LR_A.confusion(pre_y_data, ypre[1]))

LR_T.sess.close() # 不加这一句，一定会报错;加了，可能会报错。报错不影响结果

###############结果图示输出


# 绘制数据散点图
import matplotlib.pyplot as plt
def fir(datax, datay, nametotle='训练数据散点图'):
    #生成散点图需要的数据，将X数据分为2类
    datax1 = datax[datay.T[0] == 1]
    datax2 = datax[datay.T[0] == 0]

    # 散点图
    plt.scatter(datax1[:, 0], datax1[:, -1], c='r', s=28)
    plt.scatter(datax2[:, 0], datax2[:, -1], marker='^', c='b', s=28)
    plt.title(nametotle)
    plt.xlabel('X1 值')
    plt.ylabel('X2 值')

#输出分割线的表达式
def outexpre(weifgt, x=['x1', 'x2', '']):
    expression = ''
    for hh in range(len(weifgt)):
        if hh == 0:
            expression += '%s%s'%(weifgt[hh][0], x[hh])
        else:
            if weifgt[hh] > 0:
                expression += '+%s%s'%(weifgt[hh][0], x[hh])
            else:
                expression += '%s%s'%(weifgt[hh][0], x[hh])
    return expression


#三种方法生成的系数值为w1,w2,b   直线方程为w1*x1+w2*x2+b=0
#绘制三种方法各自生成的直线需要的数据
def tnd_ydata(datdxx, weights):
    dmin = datdxx[:, 0]
    minm = dmin.min()
    maxm = dmin.max()
    xda =  np.linspace(minm - 0.2, maxm + 0.2, 1000)
    yda = [(- weights[2][0] - hh * weights[0][0]) / weights[1][0] for hh in xda]
    return xda, yda


Af_data = tnd_ydata(train_x_data, lr_re.weights)
Sk_data = tnd_ydata(train_x_data, weig)
Tf_data = tnd_ydata(train_x_data, tfweib)
#首先绘制 训练数据和三种方法形成的分割线的方程
plt.plot()
fir(train_x_data, train_y_data) #绘制训练数据
plt.plot(Af_data[0], Af_data[1], '-', c='b', linewidth=1)
plt.plot(Sk_data[0], Sk_data[1], '-', c='r', linewidth=1)
plt.plot(Tf_data[0], Tf_data[1], '-', c='g', linewidth=1)
plt.title('三种方法训练的"分割线"对比')
plt.legend(['AnFany: %s'%outexpre(lr_re.weights), 'Sklearn：%s'%outexpre(weig), 'TensorFlow：%s'%outexpre(tfweib),'1类', '0类'])
plt.show()

Af_data_pre = tnd_ydata(pre_x_data, lr_re.weights)
Sk_data_pre = tnd_ydata(pre_x_data, weig)
Tf_data_pre = tnd_ydata(pre_x_data, tfweib)
#再次绘制预测数据的三种方法的表现
plt.subplot(2, 2, 1)
fir(pre_x_data, pre_y_data, nametotle='预测数据散点图') #绘制训练数据
plt.plot(Af_data_pre[0], Af_data_pre[1], '-', c='b', label='AnFany', linewidth=1)
plt.legend()


plt.subplot(2, 2, 2)
fir(pre_x_data, pre_y_data, nametotle='预测数据散点图') #绘制训练数据
plt.plot(Sk_data_pre[0], Sk_data_pre[1], '-', c='r', label='Sklearn', linewidth=1)
plt.legend()

plt.subplot(2, 2, 3)
fir(pre_x_data, pre_y_data, nametotle='预测数据散点图') #绘制训练数据
plt.plot(Tf_data_pre[0], Tf_data_pre[1], '-', c='g', label='TensorFlow', linewidth=1)
plt.legend()

plt.subplot(2, 2, 4)
plt.title('三种实现方式的预测结果的混淆矩阵对比')
plt.text(0.3, 0.6, LR_A.confusion(pre_y_data, datd))
plt.text(0.32, 0.8, 'AnFany')
plt.text(0.3, 0.3, LR_S.confusion(dm))
plt.text(0.32, 0.5, 'Sklearn')
plt.text(0.3, 0.0, LR_A.confusion(pre_y_data, ypre[1]))
plt.text(0.32, 0.2, 'TensorFlow')
plt.axis('off')



plt.show()
