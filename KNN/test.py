# -*- coding: utf-8 -*-

from kNN import KNNClassifier
from model_selection import train_test_split
import pandas as pd
import numpy as np


if __name__ == '__main__':
    """
    """
    iris = pd.read_csv('../iris.data', header=None)
    iris_data = iris.loc[:, :].values
    x_data = iris_data[:, [0, 2]]
    y_data = iris_data[:, 4]

    x_predict = np.array([[5.1, 2.1]])
    x_train, y_train, x_test, y_test = train_test_split(x_data, y_data)

    knn_clf = KNNClassifier(k=3)
    knn_clf.fit(x_train, y_train)
    y_predict = knn_clf.predict(x_test)

    print('预测准确率：{:.2f}%'.format((sum(y_predict == y_test)/len(y_test)) * 100))
