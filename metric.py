# -*- coding: utf-8 -*-

from math import sqrt
import numpy as np


def accuracy_score(y_true, y_predict):
    """计算预测准确率
    :param y_true: array
    :param y_predict: array
    :return: Float
    """
    assert y_true.shape[0] == y_predict.shape[0], 'the size of y_true must be equal to the size of y_predict'

    return sum(y_true == y_predict) / len(y_true)


def mean_squared_error(y_true, y_predict):
    """计算MSE"""
    assert len(y_true) == len(y_predict), 'the size of y_true must be equl to the size of y_predict'

    return np.sum((y_true - y_predict) ** 2) / len(y_true)


def root_mean_squared_error(y_true, y_predict):
    """计算RMSE"""
    assert len(y_true) == len(y_predict), 'the size of y_true must be equl to the size of y_predict'

    return sqrt(mean_squared_error(y_true, y_predict))


def mean_absolute_error(y_true, y_predict):
    """计算MAE"""
    assert len(y_true) == len(y_predict), 'the size of y_true must be equl to the size of y_predict'

    return np.sum(np.abs(y_true - y_predict)) / len(y_true)


def r2_score(y_true, y_predict):
    """计算Rsquare"""

    return 1 - mean_squared_error(y_true, y_predict) / np.var(y_true)
