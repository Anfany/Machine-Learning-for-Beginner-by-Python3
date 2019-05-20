#  # -*- coding：utf-8 -*-
# &Author  AnFany


#  将mnist数据集或者Fashion-MNIST转换为图片

#  因为两个数据集的格式是完全一致的，所以程序不需要修改

import struct
from PIL import Image
import numpy as np
import os

# 需要在这个文件夹子下面建立2个子文件夹mnist_train，mnist_test，分别存储训练和测试的图片数据
path = r'C:\Users\GWT9\Desktop\mnist'
os.chdir(path)


# 训练图片文件
train_images = 'train-images.idx3-ubyte'
# 训练标签文件
train_labels = 'train-labels.idx1-ubyte'

# 测试图片文件
test_images = 't10k-images.idx3-ubyte'
# 测试标签文件
test_labels = 't10k-labels.idx1-ubyte'


# 获取图片数据
def get_image(image_file):
    # 读取二进制数据
    bin_data = open(image_file, 'rb').read()
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


# 获取标签数据
def get_label(label_file):
    # 读取二进制数据
    bin_data = open(label_file, 'rb').read()
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


# 将用矩阵表示的图像信息转变为图片，名字为标签
def matrix_to_fig(matrix_data, fig_title, file_name):
    """
    将用矩阵表示的图像信息转变为图片，名字为标签
    :param matrix_data: 图像的矩阵数据
    :param fig_title: 对应的图片的标签
    :param file_name: 存储图片的文件极夹的名称
    :return: 存储图片的文件夹
    """
    sign_dict = {}
    for image, label in zip(matrix_data, fig_title):
        # 首先获取同一个标签的图片的编号，从1开始
        if label not in sign_dict:
            cc = 1
        else:
            cc = sign_dict[label] + 1
        sign_dict[label] = cc
        # 获取图片
        get_image = Image.fromarray(np.uint8(image))  # 转为uint8的格式
        # 存储图片
        get_image.save(r".\mnist_%s\%s_%d.png" % (file_name, int(label), cc))
    return print('转换完成')


# 最终的主函数

if __name__ == "__main__":
    #  获取训练图片的数字矩阵信息和标签信息
    train_fig_data = get_image(train_images)
    train_fig_label = get_label(train_labels)
    #  获取测试图片的数字矩阵信息和标签信息
    test_fig_data = get_image(test_images)
    test_fig_label = get_label(test_labels)

    # 训练数据的转换
    matrix_to_fig(train_fig_data, train_fig_label, 'train')
    # 测试数据的转换
    matrix_to_fig(test_fig_data, test_fig_label, 'test')
