# -*- coding: utf-8 -*-
import numpy as np


def train_test_split(x, y, test_ratio=0.3, seed=None):
    """将数据集按照简单交叉验证进行切割，返回分割后的结果
    :param x: 输入空间
    :param y: 输出空间
    :param test_ratio: 分割比例
    :param seed: 随机种子
    :return: x_train, y_train, x_test, y_test
    """
    assert x.shape[0] == y.shape[0], 'the size of x must be equal to the size of y'
    assert 0.0 <= test_ratio <= 1.0, 'test_ratio must be valid'

    if seed:
        np.random.seed(seed)

    # 打乱序号
    shuffled_indexes = np.random.permutation(len(x))

    test_size = int(len(x) * test_ratio)
    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]

    x_train = x[train_indexes]
    y_train = y[train_indexes]

    x_test = x[test_indexes]
    y_test = y[test_indexes]

    return x_train, y_train, x_test, y_test
