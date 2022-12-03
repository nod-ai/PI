# SharkPy

Early days of a Python -> MLIR compiler.

# Building

Just 

```shell
pip install -r requirements.txt
pip install .
```

# (Ultra) Minimal example

[test_compiler.py](./tests/test_compiler.py) (in [tests](./tests)) looks like this

```python
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
```

Running [test_compiler.py](./tests/test_compiler.py) produces (modulo some type warnings)

```mlir
module {
  func.func private @test_single_for(%arg0: f64, %arg1: f64) -> memref<10x10xf64> {
    %c10_i64 = arith.constant 10 : i64
    %0 = memref.alloca() : memref<10x10xf64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    scf.for %arg2 = %c0 to %c10 step %c1 {
      %1 = arith.mulf %arg0, %arg1 : f64
      memref.store %1, %0[%arg2, %arg2] : memref<10x10xf64>
    }
    return %0 : memref<10x10xf64>
  }
  func.func private @test_double_for(%arg0: f64, %arg1: f64) -> memref<10x10xf64> {
    %c10_i64 = arith.constant 10 : i64
    %0 = memref.alloca() : memref<10x10xf64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    scf.for %arg2 = %c0 to %c10 step %c1 {
      scf.for %arg3 = %c0 to %c10 step %c1 {
        %1 = arith.mulf %arg0, %arg1 : f64
        memref.store %1, %0[%arg2, %arg3] : memref<10x10xf64>
      }
    }
    return %0 : memref<10x10xf64>
  }
}
```

<span style="font-size:4em;">🎉</span>

# Troubleshooting

If you're having trouble installing using `pip`, try

```shell
$ pip install --no-build-isolation --no-clean . -vvvv
```

# Dev

You can load the project as a CMake project in your editor of choice with the following defines:

```cmake
-DCMAKE_PREFIX_PATH=<SOMEWHERE>/llvm_install
-DPython3_EXECUTABLE=<SOMEWHERE>/venv/bin/python
-DCMAKE_INSTALL_PREFIX=<SOMEWHERE>/install_bindings
-DMLIR_TABLEGEN_EXE=<SOMEWHERE>/llvm_install/bin/mlir-tblgen
```

Note, this won't actually build the python wheel.