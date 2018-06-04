# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
from model_selection import train_test_split
from logistic_regression import LogisticRegression


if __name__ == '__main__':
    """二分类"""
    iris = pd.read_csv('../iris.data', header=None)
    iris_data = iris.loc[:, :].values
    x_data = iris_data[:100, :2]
    y_data = iris_data[:100, 4]
    y_data[y_data == 'Iris-setosa'] = 0
    y_data[y_data == 'Iris-versicolor'] = 1

    x_train, y_train, x_test, y_test = train_test_split(x_data, y_data)
    plt.scatter(x_train[y_train == 0, 0], x_train[y_train == 0, 1], color='red')
    plt.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1], color='blue')
    plt.show()

    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)
    print(log_reg.score(x_test, y_test))
    print(y_test)
    print(log_reg.predict(x_test))
    print(log_reg.predict_proba(x_test))


