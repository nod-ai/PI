def test_single_for(a: float, b: float):
    for i in range(10):
        k = a * b
    return k


def test_double_for(a: float, b: float):
    for i in range(10):
        for j in range(10):
            k = a * b
    return k
