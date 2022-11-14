# SharkPy

# Building

You need `Graphviz`;

```shell
# Ubuntu

sudo apt-get install graphviz-dev

# Mac

brew install graphviz
```


Then just 

```shell
pip install -r requirements.txt
pip install .
```

# Tutorial

The linalg pseudo-tutorial @ `tutorial/linalg_tut.py` runs through two examples:

1. `python tutorial/linalg_tut.py --tut test` just instantiates a `linalg.matmul` and evaluates it on two inputs and prints the output (as well as evaluates the same using NumPy).
2. `python tutorial/linalg_tut.py --tut benchmark` instantiates the same `matmul` but then tiles the inner loop (using `affine-loop-tile`) and then runs and times it and prints the `mean` ∓ `variance`. You can pass `-t <TILE-SIZE>` to (naturally) change the tile size (or just run `tutorial/run_bench.sh`). Note, you need `libmlir_c_runner_utils.{so, dylib}` and `libmlir_runner_utils.{so, dylib}` for the timer functionality (by default at `<HERE>/llvm_install/lib/...`).

The script also accepts a `--debug` param, which will produce prints of the MLIR-IR at various points in the process.


## Test `matmul`

```mlir
$ python tutorial/linalg_tut.py --debug

// linalg dialect

module {
  func.func @matmul(%arg0: tensor<32x32xf64>, %arg1: tensor<32x32xf64>) -> tensor<32x32xf64> {
    %0 = linalg.init_tensor [32, 32] : tensor<32x32xf64>
    %1 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%arg0, %arg1 : tensor<32x32xf64>, tensor<32x32xf64>) outs(%0 : tensor<32x32xf64>) -> tensor<32x32xf64>
    return %1 : tensor<32x32xf64>
  }
}

...

// affine dialect

#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 2)>
module {
  func.func private @refbackend_consume_func_return_mrf64(memref<*xf64>) attributes {llvm.emit_c_interface}
  func.func @matmul(%arg0: memref<*xf64>, %arg1: memref<*xf64>) attributes {llvm.emit_c_interface} {
    %0 = memref.cast %arg0 : memref<*xf64> to memref<32x32xf64>
    %1 = memref.cast %arg1 : memref<*xf64> to memref<32x32xf64>
    %2 = memref.alloc() {alignment = 128 : i64} : memref<32x32xf64>
    %3 = memref.alloc() {alignment = 128 : i64} : memref<32x32xf64>
    affine.for %arg2 = 0 to 32 step 2 {
      affine.for %arg3 = 0 to 32 step 2 {
        affine.for %arg4 = #map0(%arg2) to #map1(%arg2) {
          affine.for %arg5 = #map0(%arg3) to #map1(%arg3) {
            %5 = affine.load %2[%arg4, %arg5] : memref<32x32xf64>
            affine.store %5, %3[%arg4, %arg5] : memref<32x32xf64>
          }
        }

...

// llvm dialect

module attributes {llvm.data_layout = ""} {
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @refbackend_consume_func_return_mrf64(%arg0: i64, %arg1: !llvm.ptr<i8>) attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i64, ptr<i8>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(i64, ptr<i8>)>
    %3 = llvm.mlir.constant(1 : index) : i64
    
...

linalg result:

[[241. 221. 232. ...  nan  nan 146.]
 [ nan  nan 196. ... 178. 161.  nan]
 [ nan  nan  nan ...  nan 155.  nan]
 ...
 [ nan  nan  nan ... 212. 187.  nan]
 [ nan  nan 251. ... 234. 198. 201.]
 [ nan 202.  nan ...  nan  nan  nan]]

NumPy result:

[[241. 221. 232. ... 212. 179. 146.]
 [177. 157. 196. ... 178. 161. 131.]
 [188. 195. 225. ... 199. 155. 166.]
 ...
 [229. 219. 238. ... 212. 187. 188.]
 [223. 212. 251. ... 234. 198. 201.]
 [219. 202. 209. ... 194. 169. 181.]]
```

## Benchmark tiled `matmul`

```shell
$ ./run_bench.sh

For tile-size 0, runtime 5700.38±29.80 ns
For tile-size 1, runtime 6922.34±77.94 ns
For tile-size 2, runtime 19329.42±52.20 ns
For tile-size 3, runtime 22820.51±46.22 ns
For tile-size 4, runtime 10334.16±51.57 ns
For tile-size 5, runtime 9200.49±77.67 ns
For tile-size 6, runtime 5409.94±27.27 ns
For tile-size 7, runtime 7098.69±107.06 ns
For tile-size 8, runtime 11157.23±52.43 ns
For tile-size 9, runtime 4907.64±31.62 ns
For tile-size 10, runtime 5593.77±178.83 ns
```

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