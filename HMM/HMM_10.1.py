# -*- coding: utf-8 -*-


import numpy as np


def HMM_10_1(A, B, Pi, T):
    """书上的算法10.1"""
    # 初始化观测序列
    O = [''] * T
    I = [0] * T
    i = 0
    # 1 根据初始状态分布选择一个盒子
    I[i] = np.random.choice(4, 1, p=Pi)[0]
    # 2
    for i in range(0, T - 1):
        # 3
        t = I[i]
        O[i] = np.random.choice(['红', '白'], 1, p=B[t])[0]
        # 4
        I[i + 1] = np.random.choice(4, 1, p=A[t])[0]
    # 5
    O[i + 1] = np.random.choice(['红', '白'], 1, p=B[t])[0]
    return O


if __name__ == '__main__':
    A = np.array([[0, 1, 0, 0],
                 [0.4, 0, 0.6, 0],
                 [0, 0.4, 0, 0.6],
                 [0, 0, 0.5, 0.5]])

    B = np.array([[0.5, 0.5],
                  [0.3, 0.7],
                  [0.6, 0.4],
                  [0.8, 0.2]])

    Pi = np.array([0.25, 0.25, 0.25, 0.25])

    print(HMM_10_1(A, B, Pi, 5))

