# Generating python stubs from pytorch

Run [get_templates_and_scripts.py](get_templates_and_scripts.py) and then run the downloaded [gen_pyi.py](gen_pyi.py).
Resulting `*.pyi` will be in [torch](torch) (i.e., right here).

# Generating pybindings for ops and `Tensor`

Just run [generate_torch_mlir_bindings.py](generate_torch_mlir_bindings.py) (note you need torch-mlir full installed).



