import numpy as np


def test_single_for(a: float, b: float):
    e = np.empty((10, 10))
    for i in range(10):
        e[i, i] = a * b
    return e


def test_double_for(a: float, b: float):
    e = np.empty((10, 10))
    for i in range(10):
        for j in range(10):
            e[i, j] = a * b
    return e


def test_mat_prod():
    A = np.empty((10, 10))
    B = np.empty((10, 10))
    C = np.empty((10, 10))
    for i in range(10):
        for j in range(10):
            C[i, j] = A[i, j] * B[i, j]
    return C


def test_mat_mul():
    A = np.empty((10, 30))
    B = np.empty((30, 20))
    C = np.empty((10, 20))
    for i in range(10):
        for j in range(30):
            for k in range(20):
                C[i, k] = A[i, j] * B[j, k]
    return C
