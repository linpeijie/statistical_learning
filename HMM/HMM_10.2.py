# -*- coding: utf-8 -*-


import numpy as np


# -*- coding: utf-8 -*-


import numpy as np


def HMM_10_2(A, B, Pi, O,T):
    """10.2 前向算法"""
    # 初始化概率P
    alpha = np.zeros(shape=[T, T])
    # 1 计算初值,下标从0开始
    for i in range(T):
        alpha[0][i] = Pi[i] * B[i][O[0]]
    # 2 计算递推公式,这里直接使用向量计算来减少计算量
    for i in range(1, T):
        for k in range(T):
            alpha[i][k] = np.dot(alpha[i - 1], A[:, k]) * B[k][O[i]]
    # 3 终止，求出最后的概率
    P = np.sum(alpha[T - 1])
    return P


if __name__ == '__main__':
    A = np.array([[0.5, 0.2, 0.3],
                 [0.3, 0.5, 0.2],
                 [0.2, 0.3, 0.5]])

    B = np.array([[0.5, 0.5],
                  [0.4, 0.6],
                  [0.7, 0.3]])

    O = np.array([0, 1, 0])

    Pi = np.array([0.2, 0.4, 0.4])

    print(HMM_10_2(A, B, Pi, O, 3))

