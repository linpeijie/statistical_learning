# -*- coding: utf-8 -*-


import numpy as np


def HMM_10_5(A, B, Pi, O, T):
    """维特比算法，DP求解最大概率路径"""
    #  导入变量
    delte = np.zeros(shape=[T, T])
    Psi = np.zeros(shape=[T, T], dtype=int)
    I = np.zeros(shape=T, dtype=int)
    # 1 初始化
    for i in range(T):
        delte[0][i] = Pi[i] * B[i][O[0]]
    # 2
    for t in range(1, T):
        for i in range(T):
            delte[t][i] = np.max(delte[t - 1] * A[:, i]) * B[i][O[t]]
            Psi[t][i] = np.argmax(delte[t - 1] * A[:, i])
    # 3 终止
    P = np.max(delte[T - 1])
    I[T - 1] = np.argmax(delte[T - 1])
    # 4 最优路径回溯
    for t in range(T - 2, -1, -1):
        I[t] = Psi[t + 1][I[t + 1]]
    return I


if __name__ == '__main__':
    A = np.array([[0.5, 0.2, 0.3],
                 [0.3, 0.5, 0.2],
                 [0.2, 0.3, 0.5]])

    B = np.array([[0.5, 0.5],
                  [0.4, 0.6],
                  [0.7, 0.3]])

    O = np.array([0, 1, 0])

    Pi = np.array([0.2, 0.4, 0.4])

    print(HMM_10_5(A, B, Pi, O, 3))
