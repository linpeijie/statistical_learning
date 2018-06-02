# -*- coding: utf-8 -*-

from decision_tree import DecisionTreeID3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    """
    table = pd.read_csv('../table_5_1.data',header=None)
    table_data = table.loc[:, :].values
    x = table_data[1:, :5]
    y = table_data[1:, 5]
    x_train, y_train, x_test, y_test = train_test_split(x, y)
    print(x_train)
    """
    table = pd.read_csv('../table_5_1.data', header=None)
    table_data = table.loc[:, :].values
    x = table_data[:, :4]
    y = table_data[:, 4]

    my_dt = DecisionTreeID3(x, y)
    my_dt.create_tree(x, y)
