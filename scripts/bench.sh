#!/usr/bin/env bash
# Runs the filter benchmark and records the environment. Run inside the dev container after
# `colcon build`:
#   docker exec gpc-dev /ros_entrypoint.sh bash /ws/src/ros2-gpu-pointcloud/scripts/bench.sh
# Writes results/bench.csv, results/stages.csv, results/bench.md and results/env.txt.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bench="${BENCH_FILTER:-/ws/install/gpu_pointcloud/lib/gpu_pointcloud/bench_filter}"
mkdir -p "$repo/results"

{
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
  echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
  echo "cuda: $(nvcc --version | grep -o 'release [0-9.]*' | cut -d' ' -f2)"
  echo "cpu: $(lscpu | grep 'Model name' | sed 's/Model name: *//')"
  echo "kernel: $(uname -r)"
  echo "base_image: $(grep -m1 '^FROM' "$repo/docker/Dockerfile" | cut -d' ' -f2)"
  echo "git_sha: $(git -c safe.directory='*' -C "$repo" rev-parse HEAD)"
  echo "git_dirty: $(git -c safe.directory='*' -C "$repo" status --porcelain -- gpu_pointcloud | wc -l) modified files under gpu_pointcloud/"
} > "$repo/results/env.txt"

"$bench" --backend all --res all --warmup 20 --iters 200 --csv "$repo/results/bench.csv" \
  --stages-csv "$repo/results/stages.csv" | tee "$repo/results/bench.md"
echo
cat "$repo/results/env.txt"
