#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Building DirectML GEMM Engine (libdml_gemm.so) ==="
g++ -O3 -shared -fPIC \
    -I"${SCRIPT_DIR}/include/wsl/stubs" \
    -I"${SCRIPT_DIR}/include" \
    "${SCRIPT_DIR}/csrc/dml_gemm_engine.cpp" \
    -ldl -o "${SCRIPT_DIR}/lib/libdml_gemm.so"

echo "✓ Build complete: ${SCRIPT_DIR}/lib/libdml_gemm.so"
