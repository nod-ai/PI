# Adding new types 

Although you probably won't have to do this, https://github.com/nod-ai/PI/pull/20/commits/d76a9c922d47ee61e72027f95287d51fe9d6434f
illustrates all these pieces to touch.

# Gotchas

If you get an error about `undefined symbol` like this

```
E   ImportError: /home/mlevental/dev_projects/PI/pi/mlir/_mlir_libs/_pi_mlir.cpython-311-x86_64-linux-gnu.so: undefined symbol: _ZN4mlir6python9PyGlobals8instanceE
```

check to make sure the relevant header `.cpp`s in `"${MLIR_INSTALL_PREFIX}/src/python/MLIRPythonExtension.Core"` were 
patched to export classes (see adjacent [cpp_ext/CMakeLists.txt:26](CMakeLists.txt)).
