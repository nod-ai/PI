# SharkPy

Early days of a Python -> MLIR compiler.

# Building

Just 

```shell
pip install -r requirements.txt
pip install .
```

# (Ultra) Minimal example

[test_kernel.py](./tests/test_kernel.py) (in [tests](./tests)) looks like this

```python
def test_kernel(a:float, b: float, c: int = 1.0, d=2.0):
    e = 0.0
    for i in range(10):
        k = a * b
    return k
```

Running [test_compiler.py](./tests/test_compiler.py) produces

```mlir
module {
  %cst = arith.constant 2.000000e+00 : f64
  func.func private @test_kernel(%arg0: f64, %arg1: f64, %arg2: si64, %arg3: f64) -> f64 {
    %cst_0 = arith.constant 0.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %cst_1 = arith.constant 0.000000e+00 : f64
    %0 = scf.for %arg4 = %c0 to %c10 step %c1 iter_args(%arg5 = %cst_0) -> (f64) {
      %1 = arith.mulf %arg0, %arg1 : f64
      scf.yield %1 : f64
    }
    return %0 : f64
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