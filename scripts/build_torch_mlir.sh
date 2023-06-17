#!/bin/bash
set -eu -o pipefail

OS=ubuntu-20.04
ARCH=AArch64
GITHUB_WORKSPACE=$PWD/..

TORCH_MLIR_MAIN_SRC_DIR=${GITHUB_WORKSPACE}/externals/torch-mlir
TORCH_MLIR_MAIN_BINARY_DIR=${GITHUB_WORKSPACE}/externals/torch-mlir/build
TORCH_MLIR_INSTALL_DIR=$PWD/torch_mlir_install/torch_mlir_install
TORCH_MLIR_HOST_MAIN_BUILD_DIR=$PWD/build_host
TORCH_MLIR_COMMIT=$(git ls-tree HEAD $TORCH_MLIR_MAIN_SRC_DIR --object-only --abbrev=8)

if [ x"$OS" == x"ubuntu-20.04" ] && [ x"$ARCH" == x"AArch64" ]; then
  cmake \
    -G Ninja \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_CXX_FLAGS="-O2 -static-libgcc -static-libstdc++" \
    -DCMAKE_C_COMPILER=gcc \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_ENABLE_TERMINFO=OFF \
    -DLLVM_ENABLE_ZLIB=OFF \
    -DLLVM_ENABLE_ZSTD=OFF \
    -DLLVM_TARGETS_TO_BUILD=X86 \
    -S${TORCH_MLIR_MAIN_SRC_DIR}/externals/llvm-project/llvm \
    -B${TORCH_MLIR_HOST_MAIN_BUILD_DIR}

    cmake --build ${TORCH_MLIR_HOST_MAIN_BUILD_DIR} \
                --target llvm-tblgen mlir-tblgen mlir-linalg-ods-yaml-gen mlir-pdll
fi

if [ x"$OS" == x"macos-latest" ]; then
  export CXX_COMPILER=clang++
  export C_COMPILER=clang
  export LLVM_DEFAULT_TARGET_TRIPLE=arm64-apple-darwin21.6.0
  export LLVM_HOST_TRIPLE=arm64-apple-darwin21.6.0
  export ARCH=AArch64
elif [ x"$OS" == x"ubuntu-20.04" ] && [ x"$ARCH" == x"AArch64" ]; then
  export CXX_COMPILER=aarch64-linux-gnu-g++
  export C_COMPILER=aarch64-linux-gnu-gcc
  export LLVM_DEFAULT_TARGET_TRIPLE=aarch64-linux-gnu
  export LLVM_HOST_TRIPLE=aarch64-linux-gnu
  export ARCH=AArch64
else
  export CXX_COMPILER=g++
  export C_COMPILER=gcc
  export LLVM_DEFAULT_TARGET_TRIPLE=x86_64-unknown-linux-gnu
  export LLVM_HOST_TRIPLE=x86_64-unknown-linux-gnu
  export ARCH=X86
fi

CMAKE_CONFIGS="\
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=$CXX_COMPILER \
  -DCMAKE_C_COMPILER=$C_COMPILER \
  -DCMAKE_INSTALL_PREFIX=$TORCH_MLIR_INSTALL_DIR \
  -DLLVM_CCACHE_BUILD=ON \
  -DLLVM_DEFAULT_TARGET_TRIPLE=$LLVM_DEFAULT_TARGET_TRIPLE \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_ENABLE_TERMINFO=OFF \
  -DLLVM_ENABLE_ZLIB=OFF \
  -DLLVM_ENABLE_ZSTD=OFF \
  -DLLVM_EXTERNAL_PROJECTS=torch-mlir;torch-mlir-dialects \
  -DLLVM_EXTERNAL_TORCH_MLIR_DIALECTS_SOURCE_DIR=${TORCH_MLIR_MAIN_SRC_DIR}/externals/llvm-external-projects/torch-mlir-dialects \
  -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR=$TORCH_MLIR_MAIN_SRC_DIR \
  -DLLVM_HOST_TRIPLE=$LLVM_HOST_TRIPLE \
  -DLLVM_INCLUDE_UTILS=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_TARGETS_TO_BUILD=$ARCH \
  -DLLVM_TARGET_ARCH=$ARCH \
  -DLLVM_USE_HOST_TOOLS=ON \
  -DMLIR_BUILD_MLIR_C_DYLIB=1 \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DMLIR_ENABLE_EXECUTION_ENGINE=ON \
  -DPython3_EXECUTABLE=$(which python) \
  -DTORCH_MLIR_ENABLE_LTC=OFF \
  -DTORCH_MLIR_ENABLE_ONLY_MLIR_PYTHON_BINDINGS=ON \
  -DTORCH_MLIR_ENABLE_STABLEHLO=OFF \
  -DTORCH_MLIR_USE_INSTALLED_PYTORCH=ON"

if [ x"$OS" == x"macos-latest" ]; then
  CMAKE_CONFIGS="${CMAKE_CONFIGS} \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DMACOSX_DEPLOYMENT_TARGET=12.0"
elif [ x"$OS" == x"ubuntu-20.04" ] && [ x"$ARCH" == x"AArch64" ]; then
  CMAKE_CONFIGS="${CMAKE_CONFIGS} \
    -DLLVM_TABLEGEN=$TORCH_MLIR_HOST_MAIN_BUILD_DIR/bin/llvm-tblgen \
    -DMLIR_LINALG_ODS_YAML_GEN=$TORCH_MLIR_HOST_MAIN_BUILD_DIR/bin/mlir-linalg-ods-yaml-gen \
    -DMLIR_LINALG_ODS_YAML_GEN_EXE=$TORCH_MLIR_HOST_MAIN_BUILD_DIR/bin/mlir-linalg-ods-yaml-gen \
    -DMLIR_PDLL_TABLEGEN=$TORCH_MLIR_HOST_MAIN_BUILD_DIR/bin/mlir-pdll \
    -DMLIR_TABLEGEN=$TORCH_MLIR_HOST_MAIN_BUILD_DIR/bin/mlir-tblgen"
fi

echo $CMAKE_CONFIGS

if [ x"$OS" == x"ubuntu-20.04" ] && [ x"$ARCH" == x"AArch64" ]; then
  cmake -G Ninja \
      $CMAKE_CONFIGS \
      -DCMAKE_CXX_FLAGS="-O2 -static-libgcc -static-libstdc++" \
      -S${TORCH_MLIR_MAIN_SRC_DIR}/externals/llvm-project/llvm \
      -B${TORCH_MLIR_MAIN_BINARY_DIR}
else
  cmake -G Ninja \
        $CMAKE_CONFIGS \
        -S${TORCH_MLIR_MAIN_SRC_DIR}/externals/llvm-project/llvm \
        -B${TORCH_MLIR_MAIN_BINARY_DIR}
fi

CMAKE_BUILD_PARALLEL_LEVEL=20 cmake --build ${TORCH_MLIR_MAIN_BINARY_DIR} --target install

OUTPUT="torch-mlir-${TORCH_MLIR_COMMIT}-$OS-$ARCH"
cd "$TORCH_MLIR_INSTALL_DIR"/..
XZ_OPT='-T0 -9' tar -cJf "${OUTPUT}.tar.xz" torch_mlir_install