# -*- coding：utf-8 -*-
# &Author  AnFany

import numpy as np

Fig_matrix = np.array([[2, 6, 3, 8, 3], [1, 4, 4, 7, 7], [3, 4, 8, 4, 9], [7, 1, 5, 4, 8], [9, 3, 1, 2, 1], [9, 5, 2, 4, 9]])

Filter_matrix = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])


# Valid卷积
def valid_c(fig, filters):
    """
    实现valid卷积
    :param fig:  图片的矩阵，numpy数组形式
    :param filters: 卷积核(滤波器)矩阵，numpy数组形式
    :return: 卷积后的结果
    """
    fig_row, fig_column = fig.shape
    filter_row, filter_column = filters.shape

    c_row = fig_row - filter_row + 1
    c_column = fig_column - filter_column + 1
    # 卷积矩阵
    c_matrix = np.zeros((c_row, c_column))

    for r in range(c_row):
        for c in range(c_column):
            matrix_f = fig[r: r + filter_row, c: c + filter_column]
            product = np.sum(matrix_f * filters)
            c_matrix[r, c] = product

    return c_matrix


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


# full卷积
def full_c(fig, filters):
    """
    实现full卷积
    :param fig:  图片的矩阵，numpy数组形式
    :param filters: 卷积核(滤波器)矩阵，numpy数组形式
    :return: 卷积后的结果
    """
    fig_row, fig_column = fig.shape
    filter_row, filter_column = filters.shape

    # 首先fig矩阵上下填充行
    row_padding = filter_row - 1
    c = np.zeros((row_padding, fig_column))
    # 矩阵上下的行填充
    fig = np.vstack((fig, c))
    fig = np.vstack((c, fig))

    # 对fig矩阵左右填充列
    column_padding = filter_column - 1
    b = np.zeros((fig_row + 2 * (filter_row - 1), column_padding))
    # 矩阵上下的行填充
    fig = np.hstack((fig, b))
    fig = np.hstack((b, fig))

    # 卷积矩阵
    c_matrix = np.zeros((fig_row + row_padding, fig_column + column_padding))

    for r in range(fig_row + row_padding):
        for c in range(fig_column + column_padding):
            matrix_f = fig[r: r + filter_row, c: c + filter_column]
            product = np.sum(matrix_f * filters)
            c_matrix[r, c] = product
    return c_matrix
