# -*- coding: utf-8 -*-
"""
Created on Mon May 21 15:35:53 2018

@author: linpeijie
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def train_data_set():
    """
    :使用《统计学习方法》习题4.1的数据作为训练数据集
    """
    x_train = np.array([[1,'S'],[1,'M'],[1,'M'],[1,'S'],[1,'S'],[2,'S'],
                        [2,'M'],[2,'M'],[2,'L'],[2,'L'],[3,'L'],[3,'M'],
                        [3,'M'],[3,'L'],[3,'L']])
    y_train = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
    
    return x_train, y_train

def test_data_set():
    """
    :测试数据集
    """
    x_test = np.array([2, 'S'])
    return x_test

def native_bayes_classifier(x_train, y_train, x_test):
    """
    :朴素贝叶斯分类器,分别计算先验概率，条件概率和后验概率，然后返回后验概率最大的类y
    :rtype: Dict
    """

    #计算Y类标记的无重复个数
    Y = np.unique(y_train)
    N = len(Y)
    #计算输入空间的样本个数
    feature = np.unique(x_train)
    #初始化先验概率和条件概率
    prior_probability = np.zeros(N)
    conditional_probability = np.zeros([N,len(feature)])
    
    #计算先验概率和条件概率
    for i in range(N):
        I_sum = np.sum(y_train == Y[i])
        prior_probability[i] = float(I_sum)/float(len(y_train))
        for j in range(len(feature)):
            feature_class_sum = np.sum((x_train[y_train == Y[i]]) == feature[j])
            conditional_probability[i][j] = float(feature_class_sum)/float(I_sum)
    
    #计算预测数据集的后验概率
    test_Y_probability = np.zeros(len(Y))
    for i in range(len(Y)):
        test_Y_probability[i] = conditional_probability[i,feature==x_test[0]] * conditional_probability[i,feature==x_test[1]] * (prior_probability[i])
        
    #结果保存为字典，并返回后验概率最大的类y
    posterior = dict(zip(Y,test_Y_probability))
    max_posterior_y = max(posterior, key=posterior.get)
    
    return max_posterior_y
    

if __name__ == '__main__':
    """
    """
    x_train, y_train = train_data_set()
    x_test = test_data_set()
    predict_y = native_bayes_classifier(x_train, y_train, x_test)  
    print('预测结果y为：{}'.format(predict_y)) #获取后验概率最大的类y，输出预测结果