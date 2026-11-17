#!/usr/bin/env bash
# Runs the filter benchmark and records the environment. Run inside the dev container after
# `colcon build --cmake-args -DWITH_PCL=ON`:
#   docker exec gpc-dev /ros_entrypoint.sh bash /ws/src/ros2-gpu-pointcloud/scripts/bench.sh
# Writes into results/: bench.csv, stages.csv, bench.md (the benchmark's stdout),
# cpu_omp_threads.csv (thread-count sweep of the OpenMP backend), cuda_launches.csv (the CUDA
# backend in separate process launches), cuda_input_modes.csv (raw upload vs host repack),
# idle_gap.csv and idle_recovery.csv (call time after idle gaps) and env.txt.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bench="${BENCH_FILTER:-/ws/install/gpu_pointcloud/lib/gpu_pointcloud/bench_filter}"
out="$repo/results"
mkdir -p "$out"

{
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
  echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
  echo "cuda: $(nvcc --version | grep -o 'release [0-9.]*' | cut -d' ' -f2)"
  echo "cpu: $(lscpu | grep 'Model name' | sed 's/Model name: *//')"
  echo "logical_cpus_visible: $(nproc)"
  echo "pcl: $(dpkg-query -W -f='${Version}' libpcl-dev 2>/dev/null || echo 'not installed')"
  echo "kernel: $(uname -r)"
  echo "base_image: $(grep -m1 '^FROM' "$repo/docker/Dockerfile" | cut -d' ' -f2)"
  echo "git_sha: $(git -c safe.directory='*' -C "$repo" rev-parse HEAD)"
  echo "git_dirty: $(git -c safe.directory='*' -C "$repo" status --porcelain -- gpu_pointcloud | wc -l) modified files under gpu_pointcloud/"
} > "$out/env.txt"

# Main run: every backend, 20 warmup + 200 timed iterations, OpenMP backend on all visible CPUs.
# Warmup also lasts at least 500 ms so a GPU that idled during the CPU runs is back at full clock.
"$bench" --backend all --res all --warmup 20 --warmup-ms 500 --iters 200 --csv "$out/bench.csv" \
  --stages-csv "$out/stages.csv" | tee "$out/bench.md"

# Thread sweep of the OpenMP backend, same protocol, so the "best CPU" claim can be checked.
echo "threads,resolution,points,backend,timing,median_ms,p99_ms,points_per_sec,out_points" > "$out/cpu_omp_threads.csv"
tmp="$(mktemp)"
for t in 1 2 4 8 12 16 20 24 32; do
  [ "$t" -le "$(nproc)" ] || continue
  "$bench" --backend cpu_omp --threads "$t" --res all --warmup 20 --warmup-ms 500 --iters 200 --csv "$tmp" > /dev/null
  tail -n +2 "$tmp" | sed "s/^/$t,/" >> "$out/cpu_omp_threads.csv"
done

# The CUDA transfer rate can differ between process launches on this platform, so the CUDA
# backend is also run in several separate launches under the same protocol.
echo "launch,resolution,points,backend,timing,median_ms,p99_ms,points_per_sec,out_points" > "$out/cuda_launches.csv"
for launch in 1 2 3 4 5 6 7; do
  "$bench" --backend cuda --res all --warmup 20 --warmup-ms 500 --iters 200 --csv "$tmp" > /dev/null
  tail -n +2 "$tmp" | sed "s/^/$launch,/" >> "$out/cuda_launches.csv"
done
rm -f "$tmp"

# Raw upload vs host repack for 12, 16 and 32 bytes per point, interleaved within each launch.
input_bench="$(dirname "$bench")/bench_cuda_input"
echo "launch,resolution,point_step,raw_ms,repack_ms,raw_h2d_ms,repack_h2d_ms" > "$out/cuda_input_modes.csv"
for launch in 1 2 3; do "$input_bench" "$launch" >> "$out/cuda_input_modes.csv"; done

# Call time when the backend was idle before the call, and recovery after 3 s of idle. The last
# run repeats the CUDA part with a tiny GPU call every 2 ms in the background.
idle_bench="$(dirname "$bench")/bench_idle_gap"
"$idle_bench" --csv "$out/idle_gap.csv" --recovery-csv "$out/idle_recovery.csv"
"$idle_bench" --backend cuda --keepalive-ms 2 --csv "$out/idle_gap_keepalive.csv" --recovery-csv /dev/null

echo
echo "OpenMP thread sweep (median ms):"
awk -F, '{print $1, $2, $6}' "$out/cpu_omp_threads.csv"
echo
echo "CUDA end_to_end median ms per launch:"
awk -F, '$5 == "end_to_end" {print $1, $2, $6}' "$out/cuda_launches.csv"
echo
echo "Median call ms by idle gap (keepalive_ms,backend,resolution,gap_ms,median_ms):"
awk -F, 'NR > 1 {print $1, $2, $3, $4, $6}' "$out/idle_gap.csv" "$out/idle_gap_keepalive.csv"
echo
cat "$out/env.txt"
