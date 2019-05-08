# -*- coding：utf-8 -*-
# &Author  AnFany


# 不同池化得到的图片对比

import pooling as p
from skimage import io
from PIL import Image
import numpy as np


def generate_fig(fig, file, func):
    """
    # 首先读取图片的矩阵，然各三个通道各自和卷积核进行卷积，将得到的结果合成为图片矩阵，最后根据矩阵显示图片
    :param fig: 需要处理的图片的路径
    :param file: 最周保存图片的路径
    :return: 图片
    """
    matrix = io.imread(fig)
    # R通道
    R = matrix[:, :, 0]
    # 计算经过same卷积后的结果

    R = func(R)
    # 通道矩阵变换维度
    R1 = R.reshape(-1, len(R[0]), 1)

    # G通道
    G = matrix[:, :, 1]
    # 计算经过same卷积后的结果
    G = func(G)
    # 通道矩阵变换维度
    G1 = G.reshape(-1, len(G[0]), 1)

    # B通道
    B = matrix[:, :, 2]
    # 计算经过same卷积后的结果
    B = func(B)
    # 通道矩阵变换维度
    B1 = B.reshape(-1, len(B[0]), 1)

    my_matrix_original = np.concatenate([R1, G1, B1], 2)

    # 将数字矩阵中的数字取整，小于0的变为0，大于255的变为255
    my_matrix_original.astype(np.int)
    my_matrix_original[my_matrix_original < 0] = 0
    my_matrix_original[my_matrix_original > 255] = 255

    # 输出图片
    image = Image.fromarray(np.uint8(my_matrix_original))  # 转换uint8格式
    image.show()
    image.save(file)

    return print('图片生成完毕')


# 最终的主函数

if __name__ == "__main__":
    # 原始图片路径
    fig_path = r'C:\Users\GWT9\Desktop\lena.jpg'
    # 池化
    func = p.Pool()
    p.p_size = 8
    p.p_strides = 10
    p.method = 3
    # 保存的文件
    save_fig = r'C:\Users\GWT9\Desktop\mean_weight.png'
    # 运行函数
    generate_fig(fig_path, save_fig, func.mean_pooling)




