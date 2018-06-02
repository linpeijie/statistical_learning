# -*- coding: utf-8 -*-

import numpy as np
from collections import Counter
import pandas as pd


class DecisionTreeID3:
    """
    """
    def __init__(self, x, y):
        """初始化
        : e 为阈值
        """
        self.x_train = x
        self.y_train = y
        self.e = 0.001

    def create_tree(self, x, y):
        """构建决策树
        :param x:
        :param y:
        :return:
        """
        # （1）若D中所有实例属于同一类C，将此类C作为节点的类标记，返回T，即创建 叶子节点
        if self.empirical_entropy(y) == 0:
            print(y)
            return y[0]
        # （2）若特征A为空集，则将剩下的所有无法再继续分类的样本点划分到含有最多样本点的类C中，创建 叶子节点
        if len(x[0]) == 1:
            return self.max_label()
        # (3)应用算法5.1 计算信息增益，选择其中信息增益最大的特征A
        feature_entropy = self.entropy(x, y)
        best_feature = np.argwhere(np.max(feature_entropy) == feature_entropy)[0][0]
        # (4)如果A的信息增益小于阈值e，将该节点设为叶子结点，且将含有最多样本点的类C作为其值，
        if best_feature < self.e:
            return self.max_label()
        # 初始化树T
        tree = {best_feature: {}}
        # (5)分割特征空间，(6)递归的调用树，直到所有情况都确定为止
        feature = x[:, best_feature]
        for value in np.unique(feature):
            x, y = self.split_feature(x, y, best_feature, value)
            tree[best_feature][value] = self.create_tree(x, y)
        return tree

    def max_label(self):
        """找出数据集X中实例数最大的类C
        :return: label
        """
        return Counter(self.y_train).most_common(1)[0][0]

    def entropy(self, x_train, y_train):
        """计算信息熵及信息增益,返回全部特征的信息增益 数组
        :param x_train: array
        :param y_train: array
        :return: List
        """
        assert x_train.shape[0] == y_train.shape[0], 'the size of X_train must be equal to the size of y_train'

        # 样本容量
        data_size = y_train.size
        # 计算经验熵 H(D)
        emp_entropy = self.empirical_entropy(y_train)
        # 计算各特征对数据集 D 的信息增益
        feature_num = x_train[0].shape[0]  # 特征 A 的个数
        info_gain = np.full(shape=feature_num, fill_value=emp_entropy)  # 初始化信息增益
        for i in range(feature_num):
            feature = x_train[:, i]  # 取出单个特征维度 A
            feature_a = np.unique(feature)  # 计算特征 A 的可能取值个数{a1,a2,a3...,an}
            for k in range(len(feature_a)):
                # 计算特征 A 对数据集D的经验条件熵
                info_gain[i] -= float(np.sum(feature_a[k] == feature))/float(data_size) \
                                * self.empirical_entropy(y_train[feature_a[k] == feature])
        print(info_gain)
        return info_gain

    def entropy_ratio(self, x_train, y_train):
        """计算信息增益比
        :param x_train: array
        :param y_train: array
        :return: List
        """
        assert x_train.shape[0] == y_train.shape[0], 'the size of X_train must be equal to the size of y_train'

        # 样本容量
        data_size = y_train.size
        # 计算经验熵 H(D)
        emp_entropy = self.empirical_entropy(y_train)
        # 计算各特征对数据集 D 的信息增益
        feature_num = x_train[0].shape[0]  # 特征 A 的个数
        feature_entropy = 1  # 初始化数据集D关于特征A的值的熵
        info_gain = np.full(shape=feature_num, fill_value=emp_entropy)  # 初始化信息增益
        info_gain_ratio = np.zeros(shape=len(info_gain))  # 初始化信息增益比
        for i in range(feature_num):
            feature = x_train[:, i]  # 取出单个特征维度 A
            feature_a = np.unique(feature)  # 计算特征 A 的可能取值个数{a1,a2,a3...,an}
            for k in range(len(feature_a)):
                # 计算特征 A 对数据集D的经验条件熵
                info_gain[i] -= float(np.sum(feature_a[k] == feature)) / float(data_size) \
                                * self.empirical_entropy(y_train[feature_a[k] == feature])
            # 计算D关于特征A的值得熵
            feature_entropy = self.empirical_entropy(feature)
            # 计算信息增益比
            info_gain_ratio[i] = info_gain[i]/feature_entropy
        print(info_gain_ratio)
        return info_gain_ratio

    @staticmethod
    def empirical_entropy(label_y):
        """计算经验熵 H(D)
        :param label_y: array
        :return: Float
        """
        assert label_y is not None, 'the label_y must be valid'

        # 统计类别数量
        counter = Counter(label_y)
        emp_entropy = 0.0
        for num in counter.values():
            p = num / len(label_y)
            emp_entropy += -p * np.log2(p)
        return emp_entropy

    def split_feature(self, x, y, index, value):
        """划分特征空间 A，返回划分后的子集
        :param x:
        :param y:
        :param index:
        :param value:
        :return:
        """
        assert  x is not None and y is not None, 'x is None, y is None'
        # 取出符合条件的样本点
        feature = x[x[:, index] == value,:]
        y = y[x[:, index] == value]
        return feature, y

    @staticmethod
    def split(x, y, d, value):
        """对特征空间进行划分
        :param x:
        :param y:
        :param d:
        :param value:
        :return:
        """
        index_a = (x[:, d] <= value)
        index_b = (x[:, d] > value)
        return x[index_a], x[index_b], y[index_a], y[index_b]

    def try_split(self, x, y):
        """寻找最优划分,使得整棵树信息熵降低，这个是计算连续性数值，并且是自动选择需要划分的特征A，而不是已经选择好了
        :param x:
        :param y:
        :return:
        """
        best_entropy = float('inf')
        best_d, best_v = -1, -1
        # 对x每个维度进行遍历,即遍历 列
        for d in range(x.shape[1]):
            sorted_index = np.argsort(x[:, d])
            for i in range(1, len(x)):  # 对每个样本进行遍历，找相邻的两个点中间的d是多少
                if x[sorted_index[i-1], d] != x[sorted_index[i], d]:
                    v = (x[sorted_index[i-1], d] + x[sorted_index[i], d]) / 2
                    x_l, x_r, y_l, y_r = self.split(x, y, d, v)
                    e = self.empirical_entropy(y_l) + self.empirical_entropy(y_r)
                    if e < best_entropy:
                        best_entropy, best_d, best_v = e, d, v
        return best_entropy, best_d, best_v
