# -*- coding：utf-8 -*-
# &Author  AnFany

from skimage import io
from PIL import Image
import numpy as np


# Same卷积
def same_c(fig, filters):
    """
    实现same卷积
    :param fig:  图片的矩阵，numpy数组形式
    :param filters: 卷积核(滤波器)矩阵，numpy数组形式
    :return: 卷积后的结果
    """
    fig_row, fig_column = fig.shape
    filter_row, filter_column = filters.shape

    # 首先fig矩阵上下填充行
    row_padding = (filter_row - 1) // 2
    c = np.zeros((row_padding, fig_column))
    # 矩阵上下的行填充
    fig = np.vstack((fig, c))
    fig = np.vstack((c, fig))

    # 对fig矩阵左右填充列
    column_padding = (filter_column - 1) // 2
    b = np.zeros((fig_row + filter_row - 1, column_padding))
    # 矩阵上下的行填充
    fig = np.hstack((fig, b))
    fig = np.hstack((b, fig))

    # 卷积矩阵
    c_matrix = np.zeros((fig_row, fig_column))

    for r in range(fig_row):
        for c in range(fig_column):
            matrix_f = fig[r: r + filter_row, c: c + filter_column]
            product = np.sum(matrix_f * filters)
            c_matrix[r, c] = product

    return c_matrix


def generate_fig(fig, c, file):
    """
    # 首先读取图片的矩阵，然各三个通道各自和卷积核进行卷积，将得到的结果合成为图片矩阵，最后根据矩阵显示图片
    :param fig: 需要处理的图片的路径
    :param c: 卷积核矩阵
    :param file: 最周保存图片的路径
    :return: 图片
    """
    matrix = io.imread(fig)
    # R通道
    R = matrix[:, :, 0]
    # 计算经过same卷积后的结果
    R = same_c(R, c)
    # 通道矩阵变换维度
    R1 = R.reshape(-1, len(R[0]), 1)

    # G通道
    G = matrix[:, :, 1]
    # 计算经过same卷积后的结果
    G = same_c(G, c)
    # 通道矩阵变换维度
    G1 = G.reshape(-1, len(G[0]), 1)

    # B通道
    B = matrix[:, :, 2]
    # 计算经过same卷积后的结果
    B = same_c(B, c)
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
    # 卷积核
    # c_matrix = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])  # 单位卷积核
    # c_matrix = 1/9 * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])  # 均值模糊
    # c_matrix = 1 / 273 * np.array([[1, 4, 7, 4, 1], [4, 16, 26, 16, 4],
    #                               [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]])  # 高斯模糊
    c_matrix = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])  # 拉普拉斯卷积核。边缘检测，锐化


    # 保存的文件
    save_fig = r'C:\Users\GWT9\Desktop\lena_laplace.png'
    # 运行函数
    generate_fig(fig_path, c_matrix, save_fig)



