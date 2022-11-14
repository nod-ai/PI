#!/bin/bash

set -e

for t in {0..20} ; do
    python linalg_tut.py -t $t --tut benchmark
done