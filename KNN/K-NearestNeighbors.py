# -*- coding: utf-8 -*-
"""
Created on Fri May 18 15:01:28 2018

@author: linpeijie
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from collections import Counter

def train_data_set():
    """
    :训练数据集
    :rtype: array
    """
    x = np.array([[3.398647362,2.331234922],
                  [3.119038289,1.783234213],
                  [1.343585456,3.364865156],
                  [3.582265486,4.679116545],
                  [2.654854654,2.565487894],
                  [7.465465465,4.676546546],
                  [5.745646465,3.565464565],
                  [9.654645423,2.511151323],
                  [7.321354876,3.420546546],
                  [7.965454654,0.789654613]])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    return x, y

def test_data_set():
    """
    :测试数据集
    :rtype: array
    """
    x_test = np.array([8.123211324,3.135468746])
    
    return x_test

def view(x_train, y_train, x_test):
    """
    ：绘制散点图
    :type x: array
    :type y: array
    :x[y == 1, 0] array的简化操作，y == 1查找出所有值为1的元素，再把这些元素的位置
                  传给x
    """
    plt.scatter(x_train[y_train == 0, 0], x_train[y_train == 0, 1], color = 'g')
    plt.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1], color = 'r')
    plt.scatter(x_test[0], x_test[1], color = 'b')
    plt.show()
    
def KNN(x_train, y_train, x_test):
    """
    :计算欧拉距离，两点间的距离
    """
    distances = [sqrt(np.sum(x_train - x_test)**2) for x_train in x_train]
    nearest = np.argsort(distances) #返回排序后的索引,list形式
    k = 6
    topK_y = [y_train[i] for i in nearest[:k]] #找出最近的K的点的标签y
    votes = Counter(topK_y) #计算数组topK_y中各元素出现的次数,找出票数最多的元素
    predict_y = votes.most_common(1)[0][0] #找出票数最多的元素,返回元组里的元素
    print("预测结果为：{}".format(predict_y))

if __name__ == '__main__':
    """
    :运行程序
    """
    x_train, y_train = train_data_set()
    x_test = test_data_set()
    view(x_train, y_train, x_test)
    KNN(x_train, y_train, x_test)