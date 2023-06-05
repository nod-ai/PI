from packaging import version


# hack purely to satisfy torch_mlir_e2e_test/test_suite/__init__.py:20
def torch_version_for_comparison():
    return version.parse("0.0.4")
