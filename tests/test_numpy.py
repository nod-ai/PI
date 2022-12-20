# from dataclasses import dataclass

from shark.dialects import memref


def test_mat_mul(a: float, b: float, c: float):
    print("inside test_mat_mul", a, b, c)
    A = memref.AllocaOp((10, 30)).memref
    B = memref.AllocaOp((30, 20)).memref
    C = memref.AllocaOp((10, 20)).memref
    for i in range(10):
        for j in range(30):
            for k in range(20):
                C[i, k] += A[i, j] * B[j, k]
    print("inside test_mat_mul", i)
    return C


if __name__ == "__main__":
    test_mat_mul(1, 2, 3)
