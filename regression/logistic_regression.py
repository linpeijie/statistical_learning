# -*- coding: utf-8 -*-

import numpy as np
from metric import accuracy_score


class LogisticRegression:

    def __init__(self):
        """初始化模型
        :coef_: w 权值向量
        :intercept_: bias 偏置
        :_theta: 0 前两者合并后的向量
        """
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    def _sigmoid(self, t):
        return 1. / (1. + np.exp(-t))

    def fit(self, x_train, y_train, eta=0.01, n_iters=1e4):
        """根据训练数据集， 使用梯度下降法训练模型
        :param x_train:
        :param y_train:
        :param eta:
        :param n_iters:
        :return:
        """
        assert x_train.shape[0] == y_train.shape[0], 'the size of x_train must be equal to the size of y_train'

        def cost(theta, x_b, y):
            """求解损失函数，这里使用的是对数似然函数
            :param theta: 求出来的参数
            :param x_b: 实例
            :param y: 类别
            :return: L(w)的值
            """
            y_hat = self._sigmoid(np.array(x_b.dot(theta), dtype=np.float32))
            try:
                return - np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat)) / len(y)
            except:
                return float('inf')

        def dcost(theta, x_b, y):
            """求梯度函数
            :param theta:
            :param x_b:
            :param y:
            :return:
            """
            return x_b.T.dot(self._sigmoid(np.array(x_b.dot(theta), dtype=np.float32)) - y) / len(x_b)

        def gradient_descent(x_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
            """应用梯度下降法求解最小值
            : n_iters: 循环次数，即要计算几个点
            """
            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = dcost(theta, x_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if (abs(cost(theta, x_b, y) - cost(last_theta, x_b, y)) < epsilon):
                    break
                cur_iter += 1
            return theta

        x_b = np.hstack([np.ones((len(x_train), 1)), x_train])  # 特征向量，1是b的内积向量
        initial_theta = np.zeros(x_b.shape[1])  # 参数向量的大小
        self._theta = gradient_descent(x_b, y_train, initial_theta, eta, n_iters)  # 模型的参数

        self.intercept_ = self._theta[0]  # 即 bias 偏置
        self.coef_ = self._theta[1:]  # 即 w 权值向量

        return self

    def predict_proba(self, x_predict):
        """给定待预测数据集x_predict, 返回表示x_predict的结果概率向量
        :param x_predict:
        :return:
        """
        assert self.intercept_ is not None and self.coef_ is not None, 'must fit before predict!'
        assert x_predict.shape[1] == len(self.coef_), 'the feature number of x_predict must be equal to x_train'

        x_b = np.hstack([np.ones((len(x_predict), 1)), x_predict])
        return self._sigmoid(np.array(x_b.dot(self._theta), dtype=np.float32))

    def predict(self, x_predict):
        """给定待测试数据集，返回表示x_predict的结果向量"""
        assert self.intercept_ is not None and self.coef_ is not None, 'must fit before predict!'
        assert x_predict.shape[1] == len(self.coef_), 'the feature number of x_predict must be equal to x_train'

        proba = self.predict_proba(x_predict)
        # True变为1，False变为0
        return np.array(proba >= 0.5, dtype='int')

    def score(self, x_test, y_test):
        """当前模型的分类准确度"""
        y_predict = self.predict(x_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "LogisticRegression()"




