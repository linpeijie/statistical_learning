# -*- coding: utf-8 -*-

import numpy as np


class LinearRegression:
    def __init__(self):
        """初始化模型
        : a_: 所求参数
        : b_:
        """
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """
        :param x_train: 训练数据集
        :param y_train:
        :return: 返回训练后的模型
        """
        assert x_train.ndim == 1, '暂时只能处理一纬的数据'
        assert len(x_train) == len(y_train), 'the size of x_train must be equal to the size of y_train'

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        denominator = 0.0
        numerator = 0.0

        for x, y in zip(x_train, y_train):
            numerator += (x - x_mean) * (y - y_mean)
            denominator += (x - x_mean) ** 2

        self.a_ = numerator / denominator
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        """给定待预测数据集，返回表示x_predict的结果向量
        :param x_predict: 向量
        :return:
        """
        assert x_predict.ndim == 1, '数据集必须得是一纬的'
        assert self.a_ is not None and self.b_ is not None, 'must fit before predict!'

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        """计算单个点的预测结果并返回
        :param x_single:
        :return:
        """
        return self.a_ * x_single + self.b_

    def __repr__(self):
        """LinearRegression()"""
