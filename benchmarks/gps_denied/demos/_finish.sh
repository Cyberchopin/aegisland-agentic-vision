#!/usr/bin/env bash
# Copy the small artifacts of HIL runs next to the chart and write SHA256SUMS.
# usage: _finish.sh <result_dir> <run_dir>...
set -euo pipefail
OUT=$1; shift
for RUN in "$@"; do
  name=$(basename "$RUN"); mkdir -p "$OUT/$name"
  cp "$RUN/summary.json" "$RUN/sent.jsonl" "$OUT/$name/"
done
(cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
echo "[result] $OUT"
