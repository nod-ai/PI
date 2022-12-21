from shark.dialects import memref
def test_mat_mul(a: float, b: float, c: float):
    # print("inside test_mat_mul", a, b, c)
    A = memref.AllocaOp((10, 30))
    B = memref.AllocaOp((30, 20))
    C = memref.AllocaOp((10, 20))
    for i in range(10):
        for j in range(30):
            for k in range(20):
                C[i, k] += A[i, j] * B[j, k] * a
                c = C[i, k]
                print(C)
    # print("inside test_mat_mul", i)
    return C


def test_if(a: float, b: float, c: float):
    # print("inside test_if", a)
    A = memref.AllocaOp((1, 1))
    d = a * b
    if a > 1:
        A[1, 1] = b
    else:
        A[1, 1] = c

    d = A[1, 1]
    return A


test_mat_mul(1, 2, 3)
test_if(1, 2, 3)
