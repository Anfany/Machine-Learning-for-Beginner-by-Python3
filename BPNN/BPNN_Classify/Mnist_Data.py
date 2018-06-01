#  # -*- coding：utf-8 -*-
# &Author  AnFany

import os
path = r'C:\Users\GWT9\Desktop'
os.chdir(path)

import numpy as np
import struct

# 训练图片文件
train_images = 'train-images.idx3-ubyte'
# 训练标签文件
train_labels = 'train-labels.idx1-ubyte'

# 测试图片文件
test_images = 't10k-images.idx3-ubyte'
# 测试标签文件
test_labels = 't10k-labels.idx1-ubyte'


def getimage(idx3_ubyte_file):
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()
    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images

def getlabel(idx1_ubyte_file):
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()
    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels

#  01化函数
def stand(data):
    futr = np.transpose(data)
    for hh in range(len(futr)):
        if np.all(futr[hh] == 0):
            pass
        else:
            futr[hh]= (futr[hh] - np.min(futr[hh])) / (np.max(futr[hh]) - np.min(futr[hh]))
    return np.transpose(futr)

#  onehot函数
def onehot(exlist, count=9):
    one = []
    for jj in exlist:
        gu = np.zeros(count + 1)
        gu[int(jj[0])] = 1
        one.append(gu)
    return np.array(one)


def run():
    trainimages = getimage(train_images)
    trainlabels = getlabel(train_labels)
    testimages = getimage(test_images)
    testlabels = getlabel(test_labels)

    # 训练数据集：随机选取5个图片和标签，看是否对应
    # import matplotlib.pyplot as plt
    # from pylab import mpl
    # mpl.rcParams['font.sans-serif'] = ['FangSong']  # 中文字体名称
    # mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
    # for hh in list(np.random.choice(list(range(len(trainimages))), 5)):
    #     plt.imshow(trainimages[hh], cmap='gray')
    #     plt.title('训练：标签为%s' % trainlabels[hh])
    #     plt.show()
    # # 测试数据集：随机选取5个图片和标签，看是否对应
    # for hh in list(np.random.choice(list(range(len(testimages))), 5)):
    #     plt.imshow(testimages[hh], cmap='gray')
    #     plt.title('测试：标签为%s' % testlabels[hh])
    #     plt.show()

    # 将表示每张图片的28*28的数组转化为1*784的数组
    Trainim = trainimages.reshape(60000, -1)
    Testim = testimages.reshape(10000, -1)

    Trainla = trainlabels.reshape(60000, -1)
    Testla = testlabels.reshape(10000, -1)

    #  输入数据标准化
    intrain, intest = stand(Trainim), stand(Testim)

    #  输出数据独热化
    outtrain, outtest = onehot(Trainla), onehot(Testla)

    return intrain, outtrain, intest, outtest


TrainIN, TrainOUT, TestIN, TestOUT = run()

# 首先将训练输入和输出合并
TRAIN = np.hstack((TrainIN, TrainOUT))


#  将训练数据平均分为n份，利用K折交叉验证计算模型最终的正确率
#  将训练数据分为训练数据和验证数据

def kfold(trdata, k=10):
    legth = len(trdata)
    datadict = {}
    signnuber = np.arange(legth)
    for hh in range(k):
        np.random.shuffle(trdata)
        yanzhneg = np.random.choice(signnuber, int(legth / k), replace=False)
        oneflod_yan = trdata[yanzhneg]
        oneflod_xun = trdata[[hdd for hdd in signnuber if hdd not in yanzhneg]]
        datadict[hh] = [oneflod_xun, oneflod_yan]
    return datadict
