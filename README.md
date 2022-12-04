
<img width="598" alt="image" src="https://user-images.githubusercontent.com/5657668/205545845-544fe701-79d5-43c1-beec-09763f22cc85.png">

# SharkPy

- [Building](#building)
- [(Ultra) Minimal example](#-ultra--minimal-example)
- [Moderately interesting example](#moderately-interesting-example)
- [Troubleshooting](#troubleshooting)
- [Development](#development)


# Building

We use [pyre](https://pyre-check.org/docs/getting-started/) for type inference. So you're going to need to install their chosen
watchdog service so that we can query for types during compilation:

```shell
brew install watchman
```

or 

```shell
sudo apt-get install watchman
```

Then `pip install pyre-check` and `pyre init` in whichever directory your 
kernel files live ([tests](./tests) has already been `init`ed).

Then just 

```shell
pip install -r requirements.txt
pip install .
```

and you're good to go.

# (Ultra) Minimal example

[test_numpy.py](./tests/test_numpy.py) (in [tests](./tests)) looks like this

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

# Moderately interesting example

[test_numpy.py](./tests/test_numpy.py) also has a matrix multiplication implementation:

```python
def test_mat_mul():
    A = np.empty((10, 30))
    B = np.empty((30, 20))
    C = np.empty((10, 20))
    for i in range(10):
        for j in range(30):
            for k in range(20):
                C[i, k] = A[i, j] * B[j, k]
    return C
```

[test_tiling.py](./tests/test_tiling.py) operates on this kernel 
but lowers it to `affine.for` loops **and** runs a few passes them, including tiling:

From this

```mlir
func.func private @test_mat_mul() -> memref<10x20xf64> {
  %0 = memref.alloca() : memref<10x30xf64>
  %1 = memref.alloca() : memref<30x20xf64>
  %2 = memref.alloca() : memref<10x20xf64>
  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to 30 {
      affine.for %arg2 = 0 to 20 {
        %3 = memref.load %0[%arg0, %arg1] : memref<10x30xf64>
        %4 = memref.load %0[%arg0, %arg1] : memref<10x30xf64>
        %5 = memref.load %1[%arg1, %arg2] : memref<30x20xf64>
        %6 = arith.mulf %4, %5 : f64
        %7 = memref.load %1[%arg1, %arg2] : memref<30x20xf64>
        memref.store %6, %2[%arg0, %arg2] : memref<10x20xf64>
      }
    }
  }
  return %2 : memref<10x20xf64>
}
```

to this

```mlir
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 2)>
func.func private @test_mat_mul() -> memref<10x20xf64> {
  %0 = memref.alloca() : memref<10x30xf64>
  %1 = memref.alloca() : memref<30x20xf64>
  %2 = memref.alloca() : memref<10x20xf64>
  affine.for %arg0 = 0 to 10 step 2 {
    affine.for %arg1 = 0 to 30 step 2 {
      affine.for %arg2 = 0 to 20 step 2 {
        affine.for %arg3 = #map0(%arg0) to #map1(%arg0) {
          affine.for %arg4 = #map0(%arg1) to #map1(%arg1) {
            affine.for %arg5 = #map0(%arg2) to #map1(%arg2) {
              %3 = memref.load %0[%arg3, %arg4] : memref<10x30xf64>
              %4 = memref.load %1[%arg4, %arg5] : memref<30x20xf64>
              %5 = arith.mulf %3, %4 : f64
              memref.store %5, %2[%arg3, %arg5] : memref<10x20xf64>
            }
          }
        }
      }
    }
  }
  return %2 : memref<10x20xf64>
}
```

# Troubleshooting

If you're having trouble installing using `pip`, try

```shell
$ pip install --no-build-isolation --no-clean . -vvvv
```

# Development

You can load the project as a CMake project in your editor of choice with the following defines:

```cmake
-DCMAKE_PREFIX_PATH=<SOMEWHERE>/llvm_install
-DPython3_EXECUTABLE=<SOMEWHERE>/venv/bin/python
-DCMAKE_INSTALL_PREFIX=<SOMEWHERE>/install_bindings
-DMLIR_TABLEGEN_EXE=<SOMEWHERE>/llvm_install/bin/mlir-tblgen
```

Note, this won't actually build the python wheel.