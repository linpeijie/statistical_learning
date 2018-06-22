# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from perceptron import Perceptron
from model_selection import train_test_split


if __name__ == '__main__':
    iris = pd.read_csv('../iris.data', header=None)
    iris_data = iris.loc[:, :].values
    x_data = iris_data[:100, [0, 2]]
    y_data = iris_data[:100, 4]
    y_data[y_data == 'Iris-setosa'] = 1
    y_data[y_data == 'Iris-versicolor'] = -1

    x_train, y_train, x_test, y_test = train_test_split(x_data, y_data)
    
    plt.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1])
    plt.scatter(x_train[y_train == -1, 0], x_train[y_train == -1, 1])
    plt.show()

    pla = Perceptron()
    pla.fit(x_train, y_train)
    print(pla.predict(x_test))
    print(pla.score(x_test, y_test))


    