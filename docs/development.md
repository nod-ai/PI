# Dev Docs

**Disclaimer 1**: I fully expect to support this project for the next 3-6 months
so while your first impulse should definitely be to try to understand the code
on your own, your second impulse should be to reach out to me (email or discord).

**Disclaimer 2**: Everything in this doc (and in this project as of today) is about
generating IR - there is nothing here about *running* code corresponding to compiled
programs that represent PyTorch models (or whatever). So any Python/PyTorch code that looks
like it's passing `Tensor`s actually isn't (it's passing IR SSA values).

**Disclaimer 3**: `torch` here refers almost always to the torch-mlir's `torch` dialect rather than `import torch`.

## Big Picture

This project is about translating Python/PyTorch models to torch-mlir IR **completely independently or upstream PyTorch**.

This is in contrast to torch-mlir itself which nominally accomplishes the same goal **but use PyTorch in various ways**.
If you want to know why you might want to avoid the PyTorch dependency jump to the [bottom](development.md#why) but the why probably isn't really important before the how.
The how is (succinctly):
1. Use torch-mlir `torch` dialect types and pybind11 to build Python wrappers for `mlir.ir.Value`s that correspond to those types.
2. Use pybind11 to wrap the `torch` dialect operation builders (which expect `mlir.ir.Value` with appropriate types) in a fully-typed manner.
3. Connect `pi.nn.modules` (which are "stolen" from upstream PyTorch) directly to the wrapped operation builders.

## Minimal example

This is a "pseudo-code"-ish minimal example for the runtime flow, i.e., not how things were designed/architected but how it works at runtime today.

```python
from pi.mlir.utils import mlir_mod_ctx, ones
from pi import nn

with mlir_mod_ctx() as module:
    x = ones(10)
    lin = nn.Linear(10, 10)
    y = lin(x)

print(module)
```

This will print

```mlir
module {
  // x
  %0 = torch.tensor.literal(dense<1.000000e+00> : tensor<10xf64>) : !torch.tensor<[10],f64>
  // bias
  %1 = torch.tensor.literal(dense<[1.105403553, ...]> : tensor<10xf64>) : !torch.tensor<[10],f64>
  // weight
  %2 = torch.tensor.literal(dense<[[-0.1630545, ...], [...], [...]]> : tensor<10x10xf64>) : !torch.tensor<[10,10],f64>
  %3 = torch.aten.linear %0, %2, %1 : !torch.tensor<[10],f64>, !torch.tensor<[10,10],f64>, !torch.tensor<[10],f64> -> !torch.tensor
}
```

The plumbing that connects the python to torch-mlir is

1. `x = ones(10)` calls [torch_dialect.NonValueTensorLiteralOp](https://github.com/nod-ai/PI/blob/3ef784bd1b5852f5c1cb84eb3e07d3735d96a909/pi/mlir/utils.py#L315)
2. `lin = nn.Linear(10, 10)` prepares two calls `torch_dialect.NonValueTensorLiteralOp` through [nn.Parameter.__init__](https://github.com/nod-ai/PI/blob/3ef784bd1b5852f5c1cb84eb3e07d3735d96a909/pi/nn/parameter.py#L30)
3. `y = lin(x)` 
   1. makes those [two calls](https://github.com/nod-ai/PI/blob/3ef784bd1b5852f5c1cb84eb3e07d3735d96a909/pi/nn/parameter.py#L64) during parameter [initialization](https://github.com/nod-ai/PI/blob/3ef784bd1b5852f5c1cb84eb3e07d3735d96a909/pi/nn/modules/module.py#L208)
   2. calls [TorchOps.pybinds::linear](https://github.com/nod-ai/PI/blob/2c0ceef1c8ac788fa7b95f483e35afb81e1bf161/cpp_ext/TorchOps.pybinds.cpp#L612) through [F.linear](https://github.com/nod-ai/PI/blob/3ef784bd1b5852f5c1cb84eb3e07d3735d96a909/pi/nn/modules/linear.py#L66)
   3. which calls the builder for `torch_dialect.AtenLinearOp` through [TorchOps.impls::linear](https://github.com/nod-ai/PI/blob/2c0ceef1c8ac788fa7b95f483e35afb81e1bf161/cpp_ext/TorchOps.impls.cpp#L1018)

The signature for `AtenLinearOp` according to the ODS in [Torch/IR/GeneratedTorchOps.td](https://github.com/makslevental/torch-mlir/blob/45c0bd76a412909ea2a813e20073965e4344f4eb/include/torch-mlir/Dialect/Torch/IR/GeneratedTorchOps.td#L4333-L4337) is

```
  let arguments = (ins
    AnyTorchTensorType:$input,
    AnyTorchTensorType:$weight,
    AnyTorchOptionalTensorType:$bias
  );
  let results = (outs
    AnyTorchTensorType:$result
  );
```

and the corresponding signature in all the pybind11 calls is 

```cpp
PyAnyTorchTensorValue linear(const PyAnyTorchTensorValue &input, 
                             const PyAnyTorchTensorValue &weight, 
                             const PyAnyTorchOptionalTensorValue &bias);
```

i.e., for all the types in the ODS it's basically a `f"Py{type}" | sed 's/Type/Value/g'`.

In summary, this is the basic template for everything going on here: torch-mlir ODS signatures inform pybind bindings and `nn.module` just plumbs into those bindings.

## Nitty-gritty details

TODO

## Dependency on torch-mlir

The mechanics of the dependency on torch-mlir are a little subtle; this repo has git submodule `externals/torch-mlir` but most often that is not "queried" for the necessary torch-mlir code.
I.e., by default you **do not need** to `git clone --recursive`, nor `git submodule update --init --recursive`.
So where does the actual torch-mlir code/source/w.e come from? 
It comes from the [releases page of pi](https://github.com/nod-ai/PI/releases?q=torch-mlir&expanded=true).

The way the project is currently configured is when you do `cmake ..` then [TorchMLIRConfig.cmake](https://github.com/nod-ai/PI/blob/cd7b6992db06f204df924c931f3aec94888f77bf/TorchMLIRConfig.cmake) will do this:

1. `git ls-tree HEAD externals/torch-mlir` to find the current commit of torch-mlir we're on (note you don't need to `--unshallow` for this to work);
2. Look for a `tar` of an already-compiled distro of torch-mlir at that commit on our releases page[^1];
   1. If there's a match, download that `tar` and untar it and set `TORCH_MLIR_INSTALL_DIR` to point to the untar directory;
   2. If there's no match, *then* `git submodule update --init --recursive` and build torch-mlir from source and `cmake --install` it to `$PI_SRC_DIR/torch_mlir_install`.

### Bumping torch-mlir

Given the above, if you want to bump the torch-mlir commit you can do it like this:

1. (Starting from `$PI_SRC_DIR`) Something like `git submodule update --init --recursive && cd externals/torch-mlir && git pull && git reset --hard <NEW_TORCH_MLIR_COMMIT>`
2. Fix things (if there's anything to fix due to API drift) 
3. Commit and send up a PR. This will automatically trigger [.github/workflows/build_torch_mlir.yml](https://github.com/nod-ai/PI/blob/cd7b6992db06f204df924c931f3aec94888f77bf/.github/workflows/build_torch_mlir.yml) (which watches for changes under `externals/*`), which will build the matching distro of torch-mlir. 

Note, **there is no circular dependency here between the CI and the distro**: when the CI runs, the same thing will happen that happens locally - `TorchMLIRConfig.cmake` will look for a distro, fail, and unshallow (and then torch-mlir will be built from source before pi is built during that CI run). Obviously this will be slow the first `n` times until after `build_torch_mlir` has completed (at which point `TorchMLIRConfig.cmake` will find the right distro).

## Why?

## Footnotes

[^1]: Obviously it doesn't really come from the releases page - the `tar` live somewhere in GitHub - but just that the releases page is a convenient indirection point to find the actual tar location.