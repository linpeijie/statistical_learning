# -*- coding: utf-8 -*-
"""
Created on Fri May  4 09:00:41 2018

@author: linpeijie
@该算法是书本上的2.1习题复现，属于感知机的原始模型
"""
from numpy import *
import numpy as np

#初始化w，b
w = [0,0]
b = 0

def dataSet():
    """
    :数据集
    :rtype: array
    :rtype: List
    """
    T = array([[3,3],[4,3],[1,1]])
    y = [1,1,-1]
    return T,y


def calMinL(x,y):
    """
    :计算损失函数的最小值
    :type x: List
    :type y: Int
    """
    global w,b
    return y * (np.dot(x,w) + b) #计算内积

def update(x,y):
    """
    :更新w，b
    :type x: List
    :type y: Int
    """
    global w,b
    for i in range(len(x)):
        w[i] += y * x[i]
    b += y
    

def perceptron(T,y):
    """
    :感知机算法
    :type T: array
    :type y: List
    """
    global w,b
    iteration = 0 #迭代次数
    findMinL = False #判断是否找到损失函数的最小值
    sampleNumber = T.shape[0] #样本点个数
    
    while(not findMinL):
        for i in range(sampleNumber):
            if calMinL(T[i],y[i]) <= 0:
                if iteration == 0:
                    print('{}    {} {}'.format(iteration,w,b))
                else:
                    print('{} X{} {} {}'.format(iteration,i+1,w,b))
                iteration += 1
                update(T[i],y[i])
                break
            elif i == sampleNumber - 1:
                print('{} X{} {} {}'.format(iteration,i+1,w,b))
                iteration += 1
                findMinL = True
                
    print('{}    {} {}'.format(iteration,w,b))
    
if __name__ == '__main__':
    T,y = dataSet()
    perceptron(T,y)