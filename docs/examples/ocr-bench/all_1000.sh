#!/bin/bash
set -e
echo "=== E4B (le plus rapide en premier) ==="
/tmp/ocrbench/run_bench_1000.sh e4b
echo "=== A4B ==="
/tmp/ocrbench/run_bench_1000.sh a4b
echo "=== DIFF ==="
/tmp/ocrbench/run_bench_1000.sh diff
echo "ALL DONE"
