#-*- coding：utf-8 -*-
# &Author  AnFany

# 引入三种方法
import Softmax_AnFany as SM_A  # 需要注释105行以后的内容
import Softmax_Sklearn as SM_S # 需要注释37行以后的内容
import Softmax_TensorFlow as SM_T # 需要注释86行以后的内容
import matplotlib.pyplot as plt
from pylab import mpl  # 作图显示中文
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 设置中文字体新宋体
mpl.rcParams['axes.unicode_minus'] = False
import numpy as np


#随机生成二元二分类的数据: 训练数据和预测数据的比例为8:2。预测数据不能重新随机生成，因为分布不同了。

x_data = np.random.random((900, 2))
y_data = []

# 获得类别数据
for dat in x_data:
    if dat[1] - 3 * dat[0] + 0.5 >= 0:
        y_data.append([1, 0, 0])
    elif dat[1] - 3 * dat[0] + 0.5 < 0 and dat[1] - 3 * dat[0] + 1.5 > 0 :
        y_data.append([0, 1, 0])
    elif dat[1] - 3 * dat[0] + 1.5 <= 0:
        y_data.append([0, 0, 1])
# 转换数据形式
y_data = np.array(y_data)

#拆分为训练数据集和预测数据集
def divided(xdata, ydata, percent=0.2):
    sign_list = list(range(len(xdata)))
    #用于测试的序号
    select_sign = sorted(np.random.choice(sign_list, int(len(xdata)*percent), replace=False))

    #用于训练的序号
    no_select_sign = [isign for isign in sign_list if isign not in select_sign]

    #测试数据
    x_predict_data = xdata[select_sign]
    y_predict_data = ydata[select_sign]#转化数据结构

    #训练数据
    x_train_data = xdata[no_select_sign]
    y_train_data = ydata[no_select_sign]#转化数据结构

    return x_train_data, y_train_data, x_predict_data, y_predict_data #训练的x，y;  测试的x，y

# 数据名称
Train_X, Train_Y, Predict_X, Predict_Y = divided(x_data, y_data)

# 绘制散点图
def fig_scatter(exdata, eydata, titl='训练数据散点图', co=['r', 'b', 'g'], marker=['o','*','^']):
    for ii in range(len(eydata[0])):
        datax = exdata[eydata[:, ii] == 1]
        plt.scatter(datax[:, 0], datax[:, -1], c=co[ii], s=50, marker=marker[ii])
    plt.title(titl)
    plt.legend(['1类', '2类', '3类'])
    plt.xlabel('X1 值')
    plt.ylabel('X2 值')


# 计算不同的方法得到的结果
# AnFany
lr_re = SM_A.LRReg()
lf = lr_re.Gradient(Train_X, Train_Y)
Pre = lr_re.predict(Predict_X)
#print('混淆矩阵：\n', SM_A.confusion(SM_A.transign(Predict_Y), Pre))


# Sklearn
regre = SM_S.sklr.fit(Train_X, SM_S.transign(Train_Y).T[0])
predata = np.array([SM_S.sklr.predict(Predict_X)]).T
#print('混淆矩阵：\n',SM_S.confusion(SM_S.transign(Predict_Y), predata))


# TensorFlow
tf_result = SM_T.trans_tf(Train_X, Train_Y, Predict_X)
#print('混淆矩阵：\n', SM_T.confusion(SM_T.transign(Predict_Y), tf_result[1]))

SM_T.sess.close()


# 输出训练的结果
# 首先绘制图
plt.subplot(2, 1, 1)
fig_scatter(Train_X, Train_Y)


plt.subplot(2, 1, 2)
plt.text(0.5, 0.9, '训练的结果展示', size='large', weight='extra bold')
plt.axis('off')
plt.text(0.00, 0.7, 'AnFany  系数\n%s'%lr_re.weights, size='large')
plt.text(0.33, 0.5, 'Sklearn  系数\n%s'%np.hstack((SM_S.sklr.coef_, np.array([SM_S.sklr.intercept_]).T)).T, size='large')
plt.text(0.66, 0.3, 'TensorFlow  系数\n%s'%np.vstack((tf_result[2], tf_result[3])), size='large')

plt.show()

# 输出预测的结果

#根据三种方法生成的系数，绘制分割线
#绘制三种方法各自生成的直线需要的数据
def tnd_ydata(datdxx, weights):
    dmin = datdxx[:, 0]
    x1da = np.linspace(datdxx[:, 0].min() - 0.2, datdxx[:, 0].max() + 0.2, 100)
    x2da = np.linspace(datdxx[:, 1].min() - 0.2, datdxx[:, 1].max() + 0.2, 100)
    X, Y = np.meshgrid(x1da, x2da)
    ydaset = []
    tw = weights.T
    for hh in range(len(tw)):
        yda = tw[hh][0] * X + tw[hh][1] * Y + tw[hh][2]
        ydaset.append(yda)
    return X, Y, ydaset


from mpl_toolkits.mplot3d import Axes3D
Af_data = tnd_ydata(Train_X, lr_re.weights)# AnFany
Sk_data = tnd_ydata(Train_X, np.hstack((SM_S.sklr.coef_, np.array([SM_S.sklr.intercept_]).T)).T) # Sklearn
Tf_data = tnd_ydata(Train_X, np.vstack((tf_result[2], tf_result[3]))) # TensorFlow

fig = plt.figure()
ax = fig.add_subplot(2, 2, 1, projection='3d')
fig_scatter(Train_X, Train_Y)
ax.plot_wireframe(Af_data[0], Af_data[1], Af_data[2][0], color='r')
ax.plot_wireframe(Af_data[0], Af_data[1], Af_data[2][1], color='b')
ax.plot_wireframe(Af_data[0], Af_data[1], Af_data[2][2], color='g')
ax.set_zlabel(r'$X_{i} \dot W$')
plt.title('AnFany训练得出的分割结果')
ax.set_yticks(np.linspace(-0.2, 1.4, 3))
ax.set_xticks(np.linspace(-0.2, 1.4, 3))
plt.legend(['1类', '2类', '3类', '1类分割面', '2类分割面', '3类分割面'], bbox_to_anchor=(1.2, 0.9))


ax = fig.add_subplot(2, 2, 2, projection='3d')
fig_scatter(Train_X, Train_Y)
ax.plot_wireframe(Sk_data[0], Sk_data[1], Sk_data[2][0], color='r')
ax.plot_wireframe(Sk_data[0], Sk_data[1], Sk_data[2][1], color='b')
ax.plot_wireframe(Sk_data[0], Sk_data[1], Sk_data[2][2], color='g')
ax.set_zlabel(r'$X_{i} \dot W$')
ax.set_yticks(np.linspace(-0.2, 1.4, 3))
ax.set_xticks(np.linspace(-0.2, 1.4, 3))
plt.title('Sklearn训练得出的分割结果')
plt.legend(['1类', '2类', '3类', '1类分割面', '2类分割面', '3类分割面'], bbox_to_anchor=(1.2, 0.9))



ax = fig.add_subplot(2, 2, 3, projection='3d')
fig_scatter(Train_X, Train_Y)
ax.plot_wireframe(Tf_data[0], Tf_data[1], Tf_data[2][0], color='r')
ax.plot_wireframe(Tf_data[0], Tf_data[1], Tf_data[2][1], color='b')
ax.plot_wireframe(Tf_data[0], Tf_data[1], Tf_data[2][2], color='g')
ax.set_zlabel(r'$X_{i} \dot W$')
ax.set_yticks(np.linspace(-0.2, 1.4, 3))
ax.set_xticks(np.linspace(-0.2, 1.4, 3))
plt.title('TensorFlow训练得出的分割结果')
plt.legend(['1类', '2类', '3类', '1类分割面', '2类分割面', '3类分割面'], bbox_to_anchor=(1.2, 0.9))



ax = fig.add_subplot(2, 2, 4)

plt.title('预测结果三种方法的混淆矩阵对比')

plt.text(0.2, 0.6, SM_A.confusion(SM_A.transign(Predict_Y), Pre))
plt.text(0.2, 0.3, SM_S.confusion(SM_S.transign(Predict_Y), predata))
plt.text(0.2, 0.0, SM_T.confusion(SM_T.transign(Predict_Y), tf_result[1]))
plt.axis('off')

plt.show()






