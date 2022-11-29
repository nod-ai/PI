import ast
import inspect

from shark._mlir_libs._mlir.ir import Context

from shark.compiler.module_builder import ModuleBuilder
from shark.compiler.types import normalize_args_kwargs


class CompilationError(Exception):
    def __init__(self, src, node):
        self.message = f"at {node.lineno}:{node.col_offset}:\n"
        self.message += "\n".join(src.split("\n")[: node.lineno])
        self.message += "\n" + " " * node.col_offset + "^"
        self.src = src
        self.node = node
        super().__init__(self.message)

    def __reduce__(self):
        # this is necessary to make CompilationError picklable
        return (type(self), (self.src, self.node))



# def build_shark_ir(fn):
#     context = Context()
#     global_scope = {}
#     fn_ast = ast.parse(inspect.getsource(fn))
#     normalized_args_kwargs = normalize_args_kwargs(fn, (), {})
#
#     module_builder = ModuleBuilder(context, global_scope)
#     try:
#         module_builder.visit(fn.parse())
#     except Exception as e:
#         node = module_builder.last_node
#         if node is None or isinstance(e, (NotImplementedError, CompilationError)):
#             raise e
#         raise CompilationError(fn.src, node) from e
#     ret = module_builder.module
#     # module takes ownership of the context
#     ret.context = context
#     return ret, module_builder



# def optimize_shark_ir(mod):
#     pm = _shark.ir.pass_manager(mod.context)
#     pm.enable_debug()
#     pm.add_inliner_pass()
#     pm.add_shark_combine_pass()
#     pm.add_canonicalizer_pass()
#     pm.add_cse_pass()
#     pm.add_licm_pass()
#     pm.run(mod)
#     return mod
#
#
# def ast_to_ttir(fn, signature, specialization, constants):
#     mod, _ = build_shark_ir(fn, signature, specialization, constants)
#     return optimize_shark_ir(mod)
#
#
# def ttir_to_ttgir(mod, num_warps, num_stages, compute_capability):
#     pm = _shark.ir.pass_manager(mod.context)
#     pm.add_convert_shark_to_sharkgpu_pass(num_warps)
#     pm.enable_debug()
#     # Convert blocked layout to mma layout for dot ops so that pipeline
#     # can get shared memory swizzled correctly.
#     pm.add_coalesce_pass()
#     pm.add_shark_gpu_combine_pass(compute_capability)
#     pm.add_sharkgpu_pipeline_pass(num_stages)
#     # Prefetch must be done after pipeline pass because pipeline pass
#     # extracts slices from the original tensor.
#     pm.add_sharkgpu_prefetch_pass()
#     pm.add_canonicalizer_pass()
#     pm.add_cse_pass()
#     pm.add_shark_gpu_combine_pass(compute_capability)
#     pm.add_licm_pass()
#     pm.add_shark_gpu_combine_pass(compute_capability)
#     pm.add_cse_pass()
#     pm.run(mod)
#     return mod
#
#
# def add_external_libs(mod, libs):
#     for name, path in libs.items():
#         if len(name) == 0 or len(path) == 0:
#             return
#     _shark.add_external_libs(mod, list(libs.keys()), list(libs.values()))
#
#
# def ttgir_to_llir(mod, extern_libs, compute_capability):
#     if extern_libs:
#         add_external_libs(mod, extern_libs)
#     return _shark.translate_shark_gpu_to_llvmir(mod, compute_capability)
#
#
# def llir_to_ptx(
#     mod: Any, compute_capability: int, ptx_version: int = None
# ) -> Tuple[str, int]:
#     """
#     Translate sharkGPU module to PTX code.
#     :param mod: a sharkGPU dialect module
#     :return:
#         - PTX code
#         - shared memory allocation size
#     """
#     if ptx_version is None:
#         _, cuda_version = path_to_ptxas()
#         ptx_version = ptx_get_version(cuda_version)
#     return _shark.translate_llvmir_to_ptx(mod, compute_capability, ptx_version)
#
#
# def ptx_to_cubin(ptx: str, compute_capability: int):
#     """
#     Compile sharkGPU module to cubin.
#     :param ptx: ptx code
#     :param compute_capability: compute capability
#     :return: str
#     """
#     ptxas, _ = path_to_ptxas()
#     return _shark.compile_ptx_to_cubin(ptx, ptxas, compute_capability)
#
#
# def ptx_get_kernel_name(ptx: str) -> str:
#     """
#     Get kernel name from PTX code.
#     This Kernel name is required when launching the kernel.
#     """
#     # There is a name mangling in PTX codegen, so the original kernel names in shark IR are not available in PTX/cubin.
#     assert ptx
#     for line in ptx.split("\n"):
#         line = line.strip()
#         if line.startswith("// .globl"):
#             return line.split()[-1]
#
#
# @functools.lru_cache
# def ptx_get_version(cuda_version) -> int:
#     """
#     Get the highest PTX version supported by the current CUDA driver.
#     """
#     assert isinstance(cuda_version, str)
#     major, minor = map(int, cuda_version.split("."))
#     version = major * 1000 + minor * 10
#     if version >= 11040:
#         return 74
#     if version >= 11030:
#         return 73
#     if version >= 11020:
#         return 72
#     if version >= 11010:
#         return 71
#     if version >= 11000:
#         return 70
#     if version >= 10020:
#         return 65
#     if version >= 10010:
#         return 64
#     if version >= 10000:
#         return 63
#     raise RuntimeError("shark only support CUDA 10.0 or higher")
#
#
# def path_to_ptxas():
#     prefixes = [
#         os.environ.get("shark_PTXAS_PATH", ""),
#         "",
#         "/usr",
#         os.environ.get("CUDA_PATH", default_cuda_dir()),
#     ]
#     for prefix in prefixes:
#         ptxas = os.path.join(prefix, "bin", "ptxas")
#         if os.path.exists(ptxas):
#             result = subprocess.check_output(
#                 [ptxas, "--version"], stderr=subprocess.STDOUT
#             )
#             if result is not None:
#                 version = re.search(
#                     r".*release (\d+\.\d+).*",
#                     result.decode("utf-8"),
#                     flags=re.MULTILINE,
#                 )
#                 if version is not None:
#                     return ptxas, version.group(1)
#     raise RuntimeError("Cannot find ptxas")
#
#
# def ty_to_cpp(ty):
#     if ty[0] == "*":
#         return "CUdeviceptr"
#     return {
#         "i1": "int32_t",
#         "i8": "int8_t",
#         "i16": "int16_t",
#         "i32": "int32_t",
#         "i64": "int64_t",
#         "u32": "uint32_t",
#         "u64": "uint64_t",
#         "fp32": "float",
#     }[ty]
#
#
# def generate_name_initializer(signature):
#     src = "int i = 0;\n"
#     tys = signature.split(",")
#     for i, ty in enumerate(tys):
#         src
#
#
# def binary_name_to_header_name(name):
#     if len(name) > 128:
#         # avoid filename too long errors (filename limit is 255)
#         name = "kernel_" + hashlib.sha256(name.encode("utf-8")).hexdigest()
#     return f"{name}.h"
#
#
# def generate_launcher(constants, signature):
#     arg_decls = ", ".join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())
#
#     def _extracted_type(ty):
#         if ty[0] == "*":
#             return "PyObject*"
#         return {
#             "i1": "int32_t",
#             "i32": "int32_t",
#             "i64": "int64_t",
#             "u32": "uint32_t",
#             "u64": "uint64_t",
#             "fp32": "float",
#             "fp64": "double",
#         }[ty]
#
#     def format_of(ty):
#         return {
#             "PyObject*": "O",
#             "float": "f",
#             "double": "d",
#             "long": "l",
#             "uint32_t": "I",
#             "int32_t": "i",
#             "uint64_t": "K",
#             "int64_t": "L",
#         }[ty]
#
#     format = "iiiiiKK" + "".join(
#         [format_of(_extracted_type(ty)) for ty in signature.values()]
#     )
#
#     # generate glue code
#     src = f"""
# #include \"cuda.h\"
# #include <Python.h>
#
# static inline void gpuAssert(CUresult code, const char *file, int line)
# {{
#    if (code != CUDA_SUCCESS)
#    {{
#       const char* prefix = "shark Error [CUDA]: ";
#       const char* str;
#       cuGetErrorString(code, &str);
#       char err[1024] = {{0}};
#       strcat(err, prefix);
#       strcat(err, str);
#       PyErr_SetString(PyExc_RuntimeError, err);
#    }}
# }}
#
# #define CUDA_CHECK(ans) {{ gpuAssert((ans), __FILE__, __LINE__); }}
#
# void _launch(int gridX, int gridY, int gridZ, int num_warps, int shared_memory, CUstream stream, CUfunction function, {arg_decls}) {{
#   void *params[] = {{ {', '.join(f"&arg{i}" for i in signature.keys() if i not in constants)} }};
#   if(gridX*gridY*gridZ > 0){{
#     CUDA_CHECK(cuLaunchKernel(function, gridX, gridY, gridZ, 32*num_warps, 1, 1, shared_memory, stream, params, 0));
#   }}
# }}
#
# static inline CUdeviceptr getPointer(PyObject *obj, int idx) {{
#   if (PyLong_Check(obj)) {{
#     return (CUdeviceptr)PyLong_AsUnsignedLongLong(obj);
#   }}
#   if (obj == Py_None) {{
#     return (CUdeviceptr)0;
#   }}
#   PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
#   if(ptr){{
#     PyObject *empty_tuple = PyTuple_New(0);
#     PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
#     Py_DECREF(empty_tuple);
#     Py_DECREF(ptr);
#     if (!PyLong_Check(ret)) {{
#       PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
#     }}
#     return (CUdeviceptr)PyLong_AsUnsignedLongLong(ret);
#   }}
#   PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
#   return (CUdeviceptr)0;
# }}
#
# static PyObject* launch(PyObject* self, PyObject* args) {{
#   int gridX, gridY, gridZ;
#   uint64_t _stream;
#   uint64_t _function;
#   int num_warps;
#   int shared_memory;
#   {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
#   if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &num_warps, &shared_memory, &_stream, &_function, {', '.join(f"&_arg{i}" for i, ty in signature.items())})) {{
#     return NULL;
#   }}
#   _launch(gridX, gridY, gridZ, num_warps, shared_memory, (CUstream)_stream, (CUfunction)_function, {', '.join(f"getPointer(_arg{i},{i})" if ty[0] == "*" else f"_arg{i}" for i, ty in signature.items())});
#   if(PyErr_Occurred()) {{
#     return NULL;
#   }}
#   // return None
#   Py_INCREF(Py_None);
#   return Py_None;
# }}
#
# static PyMethodDef ModuleMethods[] = {{
#   {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
#   {{NULL, NULL, 0, NULL}} // sentinel
# }};
#
# static struct PyModuleDef ModuleDef = {{
#   PyModuleDef_HEAD_INIT,
#   \"launcher\",
#   NULL, //documentation
#   -1, //size
#   ModuleMethods
# }};
#
# PyMODINIT_FUNC PyInit_launcher(void) {{
#   PyObject *m = PyModule_Create(&ModuleDef);
#   if(m == NULL) {{
#     return NULL;
#   }}
#   PyModule_AddFunctions(m, ModuleMethods);
#   return m;
# }}
# """
#
#     return src
#
#
# def default_cache_dir():
#     return os.path.join(os.environ["HOME"], ".shark", "cache")
#
#
# def default_cuda_dir():
#     return os.path.join("/usr", "local", "cuda")
#
#
# class CacheManager:
#     def __init__(self, key):
#         self.key = key
#         self.lock_path = None
#         # create cache directory if it doesn't exist
#         self.cache_dir = os.environ.get("shark_CACHE_DIR", default_cache_dir())
#         if self.cache_dir:
#             self.cache_dir = os.path.join(self.cache_dir, self.key)
#             self.lock_path = os.path.join(self.cache_dir, "lock")
#             os.makedirs(self.cache_dir, exist_ok=True)
#
#     def _make_path(self, filename):
#         return os.path.join(self.cache_dir, filename)
#
#     def has_file(self, filename):
#         if not self.cache_dir:
#             return False
#         return os.path.exists(self._make_path(filename))
#
#     def put(self, data, filename, binary=True):
#         if not self.cache_dir:
#             return
#         binary = isinstance(data, bytes)
#         if not binary:
#             data = str(data)
#         assert self.lock_path is not None
#         filepath = self._make_path(filename)
#         with FileLock(self.lock_path):
#             # use tempfile to be robust against program interruptions
#             mode = "wb" if binary else "w"
#             with open(filepath + ".tmp", mode) as f:
#                 f.write(data)
#             os.rename(filepath + ".tmp", filepath)
#
#
# @functools.lru_cache()
# def libcuda_dir():
#     loc = (
#         subprocess.check_output(["whereis", "libcuda.so"]).decode().strip().split()[-1]
#     )
#     return os.path.dirname(loc)
#
#
# @contextlib.contextmanager
# def quiet():
#     old_stdout, old_stderr = sys.stdout, sys.stderr
#     sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
#     try:
#         yield
#     finally:
#         sys.stdout, sys.stderr = old_stdout, old_stderr
#
#
# def _build(name, src, srcdir):
#     cuda_lib_dir = libcuda_dir()
#     cuda_path = os.environ.get("CUDA_PATH", default_cuda_dir())
#     cu_include_dir = os.path.join(cuda_path, "include")
#     suffix = sysconfig.get_config_var("EXT_SUFFIX")
#     so = os.path.join(srcdir, "{name}{suffix}".format(name=name, suffix=suffix))
#     # try to avoid setuptools if possible
#     cc = os.environ.get("CC")
#     if cc is None:
#         # TODO: support more things here.
#         clang = shutil.which("clang")
#         gcc = shutil.which("gcc")
#         cc = gcc if gcc is not None else clang
#     py_include_dir = get_paths()["include"]
#     ret = subprocess.check_call(
#         [
#             cc,
#             src,
#             "-O3",
#             f"-I{cu_include_dir}",
#             f"-I{py_include_dir}",
#             f"-I{srcdir}",
#             "-shared",
#             "-fPIC",
#             f"-L{cuda_lib_dir}",
#             "-lcuda",
#             "-o",
#             so,
#         ]
#     )
#     if ret == 0:
#         return so
#     # fallback on setuptools
#     extra_compile_args = []
#     library_dirs = [cuda_lib_dir]
#     include_dirs = [srcdir, cu_include_dir]
#     libraries = ["cuda"]
#     # extra arguments
#     extra_link_args = []
#     # create extension module
#     ext = setuptools.Extension(
#         name=name,
#         language="c",
#         sources=[src],
#         include_dirs=include_dirs,
#         extra_compile_args=extra_compile_args + ["-O3"],
#         extra_link_args=extra_link_args,
#         library_dirs=library_dirs,
#         libraries=libraries,
#     )
#     # build extension module
#     args = ["build_ext"]
#     args.append("--build-temp=" + srcdir)
#     args.append("--build-lib=" + srcdir)
#     args.append("-q")
#     args = dict(
#         name=name,
#         ext_modules=[ext],
#         script_args=args,
#     )
#     with quiet():
#         setuptools.setup(**args)
#     return so
#
#
# def make_so_cache_key(signature, constants):
#     # Get unique key for the compiled code
#     signature = {k: "ptr" if v[0] == "*" else v for k, v in signature.items()}
#     key = f"{''.join(signature.values())}{constants}"
#     key = hashlib.md5(key.encode("utf-8")).hexdigest()
#     return key
#
#
# def make_fn_cache_key(fn_hash, signature, configs, constants, num_warps, num_stages):
#     # Get unique key for the compiled code
#     get_conf_key = lambda conf: (sorted(conf.divisible_by_16), sorted(conf.equal_to_1))
#     configs_key = [get_conf_key(conf) for conf in configs]
#     key = f"{fn_hash}-{''.join(signature.values())}-{configs_key}-{constants}-{num_warps}-{num_stages}"
#     key = hashlib.md5(key.encode("utf-8")).hexdigest()
#     return key
#
#
# def read_or_execute(
#     cache_manager,
#     force_compile,
#     file_name,
#     metadata,
#     run_if_found: Callable[[str], bytes] = None,
#     run_if_not_found: Callable = None,
# ):
#     suffix = file_name.split(".")[1]
#     if not force_compile and cache_manager.has_file(file_name):
#         module = run_if_found(cache_manager._make_path(file_name))
#         data = module if isinstance(module, bytes) else str(module).encode("utf-8")
#         md5 = hashlib.md5(data).hexdigest()
#         has_changed = metadata and md5 != metadata["md5"][suffix]
#         return module, md5, has_changed, True
#     module = run_if_not_found()
#     data = module if isinstance(module, bytes) else str(module).encode("utf-8")
#     md5 = hashlib.md5(data).hexdigest()
#     cache_manager.put(data, file_name, True if isinstance(data, bytes) else data)
#     return module, md5, True, False
#
#
# def make_stub(name, signature, constants):
#     # name of files that are cached
#     so_cache_key = make_so_cache_key(signature, constants)
#     so_cache_manager = CacheManager(so_cache_key)
#     so_name = f"{name}.so"
#     # retrieve stub from cache if it exists
#     if not so_cache_manager.has_file(so_name):
#         with tempfile.TemporaryDirectory() as tmpdir:
#             src = generate_launcher(constants, signature)
#             src_path = os.path.join(tmpdir, "main.c")
#             with open(src_path, "w") as f:
#                 f.write(src)
#             so = _build(name, src_path, tmpdir)
#             with open(so, "rb") as f:
#                 so_cache_manager.put(f.read(), so_name, binary=True)
#     return so_cache_manager._make_path(so_name)
#
#
# def convert_type_repr(x):
#     match = re.search(r"!tt\.ptr<(.*)>", x)
#     if match is not None:
#         return "*" + convert_type_repr(match.group(1))
#     return x
#
#
# def make_hash(fn, **kwargs):
#     if isinstance(fn, shark.runtime.JITFunction):
#         configs = kwargs["configs"]
#         signature = kwargs["signature"]
#         constants = kwargs.get("constants", dict())
#         num_warps = kwargs.get("num_warps", 4)
#         num_stages = kwargs.get("num_stages", 3)
#         # Get unique key for the compiled code
#         get_conf_key = lambda conf: (
#             sorted(conf.divisible_by_16),
#             sorted(conf.equal_to_1),
#         )
#         configs_key = [get_conf_key(conf) for conf in configs]
#         key = f"{fn.cache_key}-{''.join(signature.values())}-{configs_key}-{constants}-{num_warps}-{num_stages}"
#         return hashlib.md5(key.encode("utf-8")).hexdigest()
#     assert isinstance(fn, str)
#     return hashlib.md5(Path(fn).read_text().encode("utf-8")).hexdigest()
#
#
# def compile(fn, **kwargs):
#     # we get the kernel, i.e. the first function generated in the module
#     # if fn is not a JITFunction, then it
#     # has to be a path to a file
#     context = _shark.ir.context()
#     asm = dict()
#     constants = kwargs.get("constants", dict())
#     if isinstance(fn, shark.runtime.JITFunction):
#         configs = kwargs.get("configs", None)
#         signature = kwargs["signature"]
#         if configs is None:
#             configs = [instance_descriptor()]
#         assert len(configs) == 1
#         kwargs["configs"] = configs
#         name = fn.__name__
#         first_stage = 0
#         if isinstance(signature, str):
#             signature = {k: v.strip() for k, v in enumerate(signature.split(","))}
#         kwargs["signature"] = signature
#     else:
#         assert isinstance(fn, str)
#         name, ir = os.path.basename(fn).split(".")
#         assert ir == "ttgir"
#         asm[ir] = _shark.ir.parse_mlir_module(fn, context)
#         function = asm[ir].get_single_function()
#         param_tys = [convert_type_repr(str(ty)) for ty in function.type.param_types()]
#         signature = {k: v for k, v in enumerate(param_tys)}
#         first_stage = 2
#
#     # cache manager
#     so_path = make_stub(name, signature, constants)
#     # create cache manager
#     fn_cache_manager = CacheManager(make_hash(fn, **kwargs))
#     # determine name and extension type of provided function
#     if isinstance(fn, shark.runtime.JITFunction):
#         name, ext = fn.__name__, "ast"
#     else:
#         name, ext = os.path.basename(fn).split(".")
#     # initialize compilation params
#     num_warps = kwargs.get("num_warps", 4)
#     num_stages = kwargs.get("num_stages", 3)
#     extern_libs = kwargs.get("extern_libs", dict())
#     device = kwargs.get("device", torch.cuda.current_device())
#     compute_capability = torch.cuda.get_device_capability(device)
#     compute_capability = compute_capability[0] * 10 + compute_capability[1]
#     # load metadata if any
#     metadata = None
#     if fn_cache_manager.has_file(f"{name}.json"):
#         with open(fn_cache_manager._make_path(f"{name}.json")) as f:
#             metadata = json.load(f)
#     else:
#         metadata = {"num_warps": num_warps, "num_stages": num_stages, "ctime": dict()}
#     # build compilation stages
#     stages = {
#         "ast": (lambda path: fn, None),
#         "ttir": (
#             lambda path: _shark.ir.parse_mlir_module(path, context),
#             lambda src: ast_to_ttir(src, signature, configs[0], constants),
#         ),
#         "ttgir": (
#             lambda path: _shark.ir.parse_mlir_module(path, context),
#             lambda src: ttir_to_ttgir(src, num_warps, num_stages, compute_capability),
#         ),
#         "llir": (
#             lambda path: Path(path).read_bytes(),
#             lambda src: ttgir_to_llir(src, extern_libs, compute_capability),
#         ),
#         "ptx": (
#             lambda path: Path(path).read_text(),
#             lambda src: llir_to_ptx(src, compute_capability),
#         ),
#         "cubin": (
#             lambda path: Path(path).read_bytes(),
#             lambda src: ptx_to_cubin(src, compute_capability),
#         ),
#     }
#     first_stage = list(stages.keys()).index(ext)
#     asm = dict()
#     module = fn
#     # run compilation pipeline  and populate metadata
#     for ir, (parse, compile) in list(stages.items())[first_stage:]:
#         path = fn_cache_manager._make_path(f"{name}.{ir}")
#         if ir == ext:
#             next_module = parse(fn)
#         elif (
#             os.path.exists(path)
#             and ir in metadata["ctime"]
#             and os.path.getctime(path) == metadata["ctime"][ir]
#         ):
#             next_module = parse(path)
#         else:
#             next_module = compile(module)
#             fn_cache_manager.put(next_module, f"{name}.{ir}")
#         if os.path.exists(path):
#             metadata["ctime"][ir] = os.path.getctime(path)
#         asm[ir] = next_module if ir == "cubin" else str(next_module)
#         if ir == "llir" and "shared" not in metadata:
#             metadata["shared"] = _shark.get_shared_memory_size(module)
#         if ir == "ptx":
#             metadata["name"] = ptx_get_kernel_name(next_module)
#         module = next_module
#     # write-back metadata
#     fn_cache_manager.put(json.dumps(metadata), f"{name}.json", binary=False)
#     # return handle to compiled kernel
#     return CompiledKernel(so_path, metadata, asm)
#
#
# class CompiledKernel:
#     def __init__(self, so_path, metadata, asm):
#         # initialize launcher
#         import importlib.util
#
#         spec = importlib.util.spec_from_file_location("launcher", so_path)
#         mod = importlib.util.module_from_spec(spec)
#         spec.loader.exec_module(mod)
#         self.c_wrapper = getattr(mod, "launch")
#         # initialize metadata
#         self.shared = metadata["shared"]
#         self.num_warps = metadata["num_warps"]
#         self.num_stages = metadata["num_stages"]
#         # initialize asm dict
#         self.asm = asm
#         device = torch.cuda.current_device()
#         global cuda_utils
#         if cuda_utils is None:
#             cuda_utils = CudaUtils()
#         mod, func, n_regs, n_spills = cuda_utils.load_binary(
#             metadata["name"], self.asm["cubin"], self.shared, device
#         )
#         self.cu_module = mod
#         self.cu_function = func
#
#     def __getitem__(self, grid):
#         def runner(*args, stream=None):
#             if stream is None:
#                 stream = torch.cuda.current_stream().cuda_stream
#             self.c_wrapper(
#                 grid[0],
#                 grid[1],
#                 grid[2],
#                 self.num_warps,
#                 self.shared,
#                 stream,
#                 self.cu_function,
#                 *args,
#             )
#
#         return runner
#
#     def get_sass(self, fun=None):
#         if "sass" in self.asm:
#             return self.asm["sass"]
#         fd, path = tempfile.mkstemp()
#         try:
#             with open(fd, "wb") as cubin:
#                 cubin.write(self.asm["cubin"])
#             self.sass = extract(path, fun)
#         finally:
#             os.remove(path)
#         self.asm["sass"] = self.sass
#         return self.sass
