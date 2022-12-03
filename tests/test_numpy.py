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
