#!/bin/bash

PATH="$PATH_":$PATH

echo "$(pwd)"
echo "$(ninja --version)"
echo "$(cmake --version)"
echo "$(python3 --version)"
echo "$(clang --version)"
echo "$(clang++ --version)"
