# -*- coding: utf-8 -*-

from kNN import KNNClassifier
import pandas as pd
import numpy as np


def data_set():
    """
    :鸢尾花数据集,返回处理后的数据
    :rtype: ndarray
    """
    file = '../iris.data'
    df = pd.read_csv(file, header=None)
    # 处理数据集，转换成array格式
    y = df.loc[0:, 4].values
    x = df.loc[0:, [0, 2]].values

    return x, y


if __name__ == '__main__':
    """
    """
    x_predict = np.array([[5.1, 2.1]])
    x_train, y_train = data_set()

    knn_clf = KNNClassifier(k=6)
    knn_clf.fit(x_train, y_train)
    y_predict = knn_clf.predict(x_predict)

    print('预测结果为：{}'.format(y_predict))
