# -*- coding: utf-8 -*-

import numpy as np
from math import sqrt
from collections import Counter


class KNNClassifier:
    """
    """
    def __init__(self, k):
        """
        初始化kNN分类器
        :param k: 参数
        """
        assert k >= 1, "k must be valid"
        self.k = k
        self._x_train = None
        self._y_train = None

    def fit(self, x_train, y_train):
        """
        根据训练数据集训练kNN分类器，即拟合
        :param x_train: array
        :param y_train: array
        :return: self
        """
        assert x_train.shape[0] == y_train.shape[0], 'the size of X_train must be equal to the size of y_train'
        assert self.k <= x_train.shape[0], 'the size of X_train must be at least k.'

        self._x_train = x_train
        self._y_train = y_train
        # 返回自身
        return self

    def predict(self, x_predict):
        """给定待预测数据集，返回表示x_predict的结果向量
        :param x_predict: array
        :return: array
        """
        # 将x_predict化为向量
        if len(x_predict.shape) == 1:
            x_predict = np.array([x_predict])

        assert self._x_train is not None and self._y_train is not None,\
            'must fit before predict!'
        # x_predict的列数必须等于x_train的列数，即特征数
        assert x_predict.shape[1] == self._x_train.shape[1],\
            'the feature number of x_predict must be equal to x_train'

        y_predict = [self._predict(x) for x in x_predict]
        return np.array(y_predict)

    def _predict(self, x):
        """对单个待预测数据x，返回x的预测标签y,即计算欧拉距离，返回票数最多的标签
        :param x:
        :return: label
        """
        assert x.shape[0] == self._x_train.shape[1], 'the feature number of x must be equal to x_train'

        distances = [sqrt(np.sum((x_train - x) ** 2))
                     for x_train in self._x_train]
        nearest = np.argsort(distances)

        top_k_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(top_k_y)

        return votes.most_common(1)[0][0]

    def __repr__(self):
        """
        :return: String
        """
        return "KNN(k=%d) 运行良好" % self.k
