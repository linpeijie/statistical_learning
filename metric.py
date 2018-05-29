# -*- coding: utf-8 -*-


def accuracy_score(y_true, y_predict):
    """计算预测准确率
    :param y_true: array
    :param y_predict: array
    :return: Float
    """
    assert y_true.shape[0] == y_predict.shape[0], 'the size of y_true must be equal to the size of y_predict'

    return sum(y_true == y_predict) / len(y_true)
