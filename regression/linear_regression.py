# -*- coding: utf-8 -*-

import numpy as np
from metric import r2_score


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

        # 通过循环计算分子和分母
        for x, y in zip(x_train, y_train):
            numerator += (x - x_mean) * (y - y_mean)
            denominator += (x - x_mean) ** 2

        """向量化运算,将损失函数化简，通过向量化运算来提高模型 性能
        numerator = (x_train - x_mean).dot(y_train - y_mean)
        denominator = (x_train - x_mean).dot(x_train - x_mean)
        """

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

    def score(self, x_test, y_test):
        """评估模型准确度,使用R2标准"""
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        """LinearRegression()"""


class MultipleLinearRegression:
    def __init__(self):
        """"""
        self.coef_ = None
        self.bias_ = None
        self._theta = None

    def fit_normal(self, X_train, y_train):
        """"""
        assert X_train.shape[0] == y_train.shape[0], 'the size of X_train must be equal to the size of y_train'

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.bias_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict(self, X_predict):
        """"""
        assert self.bias_ is not None and self.coef_ is not None, 'must fit before predict ！'
        assert X_predict.shape[1] == len(self.coef_)

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        """"""
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        """MultipleLinearRegression()"""
