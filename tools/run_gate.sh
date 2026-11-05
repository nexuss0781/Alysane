#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j2 >/dev/null
./build/alysane selftest
python3 tools/prepare_data.py
mkdir -p reports
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 ./build/alysane parity models data/parity.bin reports/parity_cpp.bin
OPENBLAS_NUM_THREADS=2 python3 tools/parity_reference.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 ./build/alysane gates models data reports
python3 tools/make_report.py
