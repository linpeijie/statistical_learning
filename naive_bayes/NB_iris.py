# -*- coding: utf-8 -*-
"""
Created on Tue May 22 12:29:19 2018

@author: linpeijie
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def data_set():
    """
    :采用简单交叉验证法，将原始数据集分为strain_data和test_data,分别占70%和30%
    :rtype: array
    """
    data_file = '../iris.data'
    data_set = pd.read_csv(data_file, header=None)
    data_set = data_set.loc[:,:].values
    #随机打乱原始数据集,
    np.random.shuffle(data_set)
    train_num = int(0.7 * len(data_set))
    train_data, test_data = data_set[:train_num], data_set[train_num:]
    
    return train_data, test_data

def precess_data(data):
    """
    :从数据集中分类出输入空间和输出空间
    :rtype: array
    """
    x = data[:,0:4]
    y = data[:,4]
    
    return x, y

def laplace(x_train):
    """
    :求每个维度特征可能的取值个数S
    :rtype : Dict
    """
    S = {}
    for i in range(len(x_train[0])):
        element = x_train[:,i]
        number = len(np.unique(element))
        S.update(dict.fromkeys(element,number))
    
    return S

def NBC_model(train_set, k):
    """
    :采用拉普拉斯平滑的朴素贝叶斯分类器，对训练数据集进行训练，得出模型
    :rtype: array
    """
    global feature, Y
    #处理数据集
    x_train, y_train = precess_data(train_set)
    #计算Y类标记的无重复个数
    Y = np.unique(y_train)
    N = len(y_train)
    #计算X特征无重复个数,以及拉普拉斯平滑中的S
    feature = np.unique(x_train)
    S = laplace(x_train)    
     #初始化先验概率和条件概率
    prior_probability = np.zeros(len(Y))
    conditional_probability = np.zeros([len(Y),len(feature)])
    #计算先验概率和条件概率
    for i in range(len(Y)):
        I_sum = np.sum(y_train == Y[i])
        prior_probability[i] = float(I_sum + k)/float(N + len(Y) * k)
        for j in range(len(feature)):
            feature_class_sum = np.sum((x_train[y_train == Y[i]]) == feature[j])
            conditional_probability[i][j] = float(feature_class_sum + k)/float(I_sum + S[feature[j]] * k)

    return prior_probability, conditional_probability

def NBC_predict(test_set, prior_probability, conditional_probability):
    """
    :计算后验概率，返回最大后验概率
    :rtype: array
    """
    global feature, Y
    #处理数据
    x_test, y_test = precess_data(test_set)
    #测试集样本个数
    y_predict = ['']*len(y_test)
    #计算预测数据集的后验概率
    for i in range(len(x_test)):
        try:
            test_Y_probability = np.zeros(len(Y))
            for j in range(len(Y)):
                test_Y_probability[j] = conditional_probability[j,feature==x_test[i][0]] * conditional_probability[j,feature==x_test[i][1]] * conditional_probability[j,feature==x_test[i][2]] * conditional_probability[j,feature==x_test[i][3]] * (prior_probability[j])
            #结果保存为字典，并返回后验概率最大的类y
            posterior = dict(zip(Y,test_Y_probability))
            max_posterior_y = max(posterior, key=posterior.get)
            y_predict[i] = max_posterior_y
        except ValueError:
            """
            因为测试数据集的输入空间存在模型里没有的特征，所以会出现 setting an array element with a sequence. 错误
            因为找不到相匹配的特征概率，导致数组维度不匹配
            因为这个错误，导致正确率一直低于80%
            """
            #print(feature==x_test[i][0])
        finally:
            y_predict[i] == ''
    
    return y_predict == y_test
    
if __name__ == '__main__':
    train_data, test_data = data_set()
    prior, conditional = NBC_model(train_data, 1)
    y_predict = NBC_predict(test_data, prior, conditional)
    print('预测正确率：{:.2f}%'.format(float(str(y_predict).count('True')) / float(len(y_predict)) * 100))
    