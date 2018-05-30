import numpy as np
import pandas as pd


class DecisionTreeID3:
    """
    """
    def __init__(self):
        """"""

    def entropy(self, x_train, y_train):
        """计算信息熵及信息增益
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

    @staticmethod
    def empirical_entropy(label_y):
        """计算经验熵 H(D)
        :param label_y: array
        :return: Float
        """
        assert label_y is not None, 'the label_y must be valid'

        emp_entropy = 0
        # 样本容量
        data_size = label_y.size
        # 计算类的可能取值元素
        label = np.unique(label_y)
        # 计算经验熵
        for k in range(len(label)):
            e = float(np.sum(label[k] == label_y)) / float(data_size)
            emp_entropy += -(e * np.log2(e))
        return emp_entropy