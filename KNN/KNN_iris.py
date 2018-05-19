# -*- coding: utf-8 -*-
"""
Created on Sat May 19 15:22:21 2018

@author: linpeijie
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from collections import Counter

def data_set():
    """
    :鸢尾花数据集,返回处理后的数据
    :Sepal.Length in cm
    :Sepal.Width in cm
    :Petal.Length in cm
    :Petal.Width in cm
    :class: 
      -- Iris Setosa
      -- Iris Versicolour
      -- Iris Virginica
    :rtype: ndarray
    """
    file = '../iris.data'
    df = pd.read_csv(file, header=None)
    #处理数据集，转换成ndarray格式
    y = df.loc[0:, 4].values
    x = df.loc[0:, [0,2]].values
    
    return x, y

def test_data_set():
    """
    :测试数据集
    :rtype: array
    """
    x_test = np.array([5.1,2.1])
    
    return x_test

def view(x, y, x_test, predict_y):
    """
    :绘制散点图
    """
    plt.scatter(x[y=='Iris-setosa',0],x[y=='Iris-setosa',1], color='r', marker='o', label='setosa')
    plt.scatter(x[y=='Iris-versicolor',0],x[y=='Iris-versicolor',1], color='b', marker='x', label='versicolor')
    plt.scatter(x[y=='Iris-virginica',0],x[y=='Iris-virginica',1], color='g', marker='o', label='virginica')
    plt.scatter(x_test[0], x_test[1], color='y', marker='x', label='{}'.format(predict_y))
    plt.xlabel('Sepal.Length')
    plt.ylabel('Petal.Lidth')
    plt.legend(loc = 'upper left')
    plt.show()
    
def KNN(k, x_train, y_train, x_test):
    """
    :计算欧拉距离，两点间的距离,返回预测结果
    :rtype: String
    """
    distances = [sqrt(np.sum(x_train - x_test)**2) for x_train in x_train]
    nearest = np.argsort(distances) #返回排序后的索引,list形式
    topK_y = [y_train[i] for i in nearest[:k]] #找出最近的K的点的标签y
    votes = Counter(topK_y) #计算数组topK_y中各元素出现的次数,找出票数最多的元素
    predict_y = votes.most_common(1)[0][0] #找出票数最多的元素,返回元组里的元素
    
    return predict_y
     
if __name__ == '__main__':
    """
    """
    x_train, y_train = data_set()
    x_test = test_data_set()
    predict_y = KNN(6, x_train, y_train, x_test)
    view(x_train, y_train, x_test, predict_y)
    print('预测结果是：{}'.format(predict_y))