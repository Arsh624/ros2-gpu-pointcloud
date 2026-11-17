#!/usr/bin/env bash
# Profiles the CUDA filter with Nsight Systems. Run inside the dev container after `colcon build`:
#   docker exec gpc-dev /ros_entrypoint.sh bash /ws/src/ros2-gpu-pointcloud/scripts/profile.sh
# Writes results/profile.nsys-rep (not committed) and results/nsys_summary.txt.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bench="${BENCH_FILTER:-/ws/install/gpu_pointcloud/lib/gpu_pointcloud/bench_filter}"
mkdir -p "$repo/results"
cd "$repo/results"

nsys profile -t cuda,nvtx --force-overwrite true -o profile \
  "$bench" --backend cuda --res 640x480 --iters 50

{
  echo "# nsys stats for: bench_filter --backend cuda --res 640x480 --iters 50 (20 warmup iterations included)"
  nsys stats --force-export true --report nvtx_sum --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum \
    --report cuda_api_sum profile.nsys-rep
} > nsys_summary.txt
echo "wrote $repo/results/nsys_summary.txt"
