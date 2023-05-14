# import re
#
# f = open(
#     "/home/mlevental/dev_projects/torch-mlir/install/include/torch-mlir-c/TorchTypes.h"
# ).read()
#
# matches = re.findall(r"torchMlirTypeIsA(\w+)\(", f)
#
# for ff in matches:
#     print(f"{ff}Type, {ff}Value, ")
#
# print()
#
# for ff in matches:
#     print(
#         f'''mlir_type_subclass(m, "{ff}Type", torchMlirTypeIsA{ff}).def_staticmethod("get", [](DefaultingPyMlirContext context) {{ return torchMlir{ff}TypeGet(context->get()); }}, py::arg("context") = py::none());'''
#     )
#
# print()
#
# for ff in matches:
#     print(
#         f'(void)mlir_value_subclass(m, "{ff}Value", [](MlirValue value) {{ return torchMlirTypeIsA{ff}(mlirValueGetType(value)); }});'
#     )
#
#
# g = open("/home/mlevental/.config/JetBrains/CLion2023.1/scratches/scratch_155.cpp").readlines()
# for gg in g:
#     gg = gg.strip()
#     print(f"""
# t = Torch{gg}Type.get()
# assert str(t) == '!torch.{gg.lower()}'
# assert Torch{gg}Type.isinstance(t)""")
#
#
# for ff in matches:
#     print(f"_({ff}Value) \\ ")


# from torch_mlir.dialects.torch.importer.jit_ir.build_tools.torch_ods_gen import (
#     TORCH_TYPE_TO_ODS_TYPE,
# )
#
# for k, v in sorted(TORCH_TYPE_TO_ODS_TYPE.items(), key=lambda x: x[1]):
#     v = v.replace("Type", "")
#     print(f"""
# bool isA{v}Value(MlirValue value) {{
#   return isA{v}Type(mlirValueGetType(value));
# }}""")