# -*- coding: utf-8 -*-
"""
Created on Fri May  4 09:00:41 2018

@author: linpeijie
@该算法是书本上的2.1习题复现，属于感知机的原始模型
"""
import numpy as np
from metric import accuracy_score


class Perceptron:
    def __init__(self):
        """初始化"""
        self.w_ = [0, 0]
        self.b_ = 0
        self._n = 1

    def fit(self, X_train, y_train):
        """训练模型"""
        assert X_train.shape[0] == len(y_train), 'The size of X_train must be equal to the size of y_train'
        assert X_train.shape[1] == 2, 'the feature of X_train must be equal to 2'

        find_min = False
        sample_num = len(y_train)

        while(not find_min):
            for i in range(sample_num):
                if y_train[i] * (np.dot(self.w_, X_train[i])) <= 0:
                    self.w_ = self.w_ + self._n * y_train[i] * X_train[i]
                    self.b_ = self.b_ + self._n * y_train[i]
                    find_min = False
                    break
                elif i == sample_num - 1:
                    find_min = True
    
    def predict(self, x_test):
        """预测"""
        result = [self._predict(x) for x in x_test]
        
        return result
    
    def _predict(self, x):
        """预测样本类别"""
        if np.dot(self.w_, x) + self.b_ >= 0:
            return 1
        else:
            return -1
    
    def score(self, x_test, y_test):
        """算法准确率"""
        y_predict = self.predict(x_test)
        y_predict = np.array(y_predict)

        return accuracy_score(y_test, y_predict)

