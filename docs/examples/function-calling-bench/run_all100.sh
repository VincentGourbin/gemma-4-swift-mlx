#!/bin/bash
set -e
cd /tmp/bfcl
for kind in e4b a4b diff; do
  echo "=== $kind ==="
  python3 run_bfcl100.py $kind 2>&1
  echo ""
done
echo "ALL DONE"
