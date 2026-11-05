#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
make c3v2_1
./build/c3v2_1_test | tee reports/c3v2_1_tests.txt
./build/c3v2_1_demo reports/c3v2_1_demo.json
echo "C3v2-1 complete: reports/c3v2_1_tests.txt reports/c3v2_1_demo.json"
