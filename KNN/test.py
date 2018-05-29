# -*- coding: utf-8 -*-

from KNN import KNNClassifier
from model_selection import train_test_split
from metric import accuracy_score
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

    best_score = 0.0
    best_k = -1
    for k in range(1, 11):
        knn_clf = KNNClassifier(n_neighbors=k)
        knn_clf.fit(x_train, y_train)
        score = knn_clf.score(x_test, y_test)
        if score > best_score:
            best_k = k
            best_score = score

    print("best_k =", best_k)
    print('best_score =', best_score)
