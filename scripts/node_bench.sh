#!/usr/bin/env bash
# Node-level benchmark: runs pointcloud_filter_node against cloud_publisher_node in this container
# (two processes, default RMW) and measures, per backend and resolution:
#   1. callback latency at 10 Hz and 30 Hz  -> results/node_latency.csv
#   2. what happens as the input rate rises -> results/node_rate.csv
# Run inside the dev container after `colcon build`:
#   docker exec gpc-dev /ros_entrypoint.sh bash /ws/src/ros2-gpu-pointcloud/scripts/node_bench.sh
#
# A backend is written NAME or NAME:THREADS; THREADS sets the node's cpu_threads parameter and
# only matters for cpu_omp. The OpenMP backend is swept over thread counts because inside a node,
# next to a publisher and the middleware's own threads, "all cores" is not the fastest setting.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bin="${GPC_BIN:-/ws/install/gpu_pointcloud/lib/gpu_pointcloud}"
out="$repo/results"
mkdir -p "$out"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

latency_backends="${LATENCY_BACKENDS:-cpu cpu_omp:8 cpu_omp:12 cpu_omp:16 cpu_omp:20 cuda}"
rate_backends="${RATE_BACKENDS:-cpu cpu_omp:12 cpu_omp:20 cuda}"
resolutions="${RESOLUTIONS:-320x240 640x480 1280x720}"
skip=20   # messages dropped from the start of every latency log (start-up)

# pct FILE COLUMN P SKIP: nearest-rank percentile of a CSV column, ignoring the header and SKIP rows.
pct() {
  tail -n +$((2 + $4)) "$1" | cut -d, -f"$2" | sort -g |
    awk -v p="$3" '{v[NR] = $1} END {if (NR == 0) {print "nan"; exit} i = int(p * NR + 0.999999); if (i < 1) i = 1; print v[i]}'
}

# run BACKEND[:THREADS] WxH RATE COUNT: one node + one publisher. Leaves $tmp/lat.csv, $tmp/pub.log.
run() {
  local backend="${1%%:*}" threads=0 res="$2" rate="$3" count="$4"
  [[ "$1" == *:* ]] && threads="${1##*:}"
  rm -f "$tmp/lat.csv" "$tmp/pub.log"
  "$bin/pointcloud_filter_node" --ros-args -p backend:="$backend" -p cpu_threads:="$threads" \
    -p latency_csv:="$tmp/lat.csv" > "$tmp/node.log" 2>&1 &
  local node_pid=$!
  "$bin/cloud_publisher_node" --ros-args -r cloud:=input -p width:="${res%x*}" -p height:="${res#*x}" \
    -p rate:="$rate.0" -p count:="$count" -p wait_for_subscribers:=1 > "$tmp/pub.log" 2>&1
  sleep 0.5   # let the node finish the last messages
  kill -INT "$node_pid"
  wait "$node_pid" || true
}

echo "backend,resolution,rate_hz,messages,transport_p50_ms,view_p50_ms,filter_p50_ms,publish_p50_ms,total_p50_ms,total_p99_ms,total_max_ms" \
  > "$out/node_latency.csv"
for backend in $latency_backends; do
  for res in $resolutions; do
    for rate in 10 30; do
      run "$backend" "$res" "$rate" $((rate * 10 + skip))
      f="$tmp/lat.csv"
      n=$(($(wc -l < "$f") - 1 - skip))
      echo "${backend/:/_},$res,$rate,$n,$(pct "$f" 3 0.5 $skip),$(pct "$f" 4 0.5 $skip),$(pct "$f" 5 0.5 $skip),$(pct "$f" 6 0.5 $skip),$(pct "$f" 7 0.5 $skip),$(pct "$f" 7 0.99 $skip),$(pct "$f" 7 1.0 $skip)" \
        | tee -a "$out/node_latency.csv"
    done
  done
done

# Rising input rate, 4 s per step. "received" is what reached the node's callback.
echo "backend,resolution,requested_hz,achieved_hz,sent,received,drop_pct,total_p50_ms" > "$out/node_rate.csv"
for backend in $rate_backends; do
  for res in $resolutions; do
    for rate in 60 120 240 480; do
      run "$backend" "$res" "$rate" $((rate * 4))
      sent=$(grep -o 'sent=[0-9]*' "$tmp/pub.log" | cut -d= -f2)
      achieved=$(grep -o 'achieved_hz=[0-9.]*' "$tmp/pub.log" | cut -d= -f2)
      received=$(($(wc -l < "$tmp/lat.csv") - 1))
      drop=$(awk -v s="$sent" -v r="$received" 'BEGIN {d = 0; if (s > 0) d = 100 * (s - r) / s; printf "%.1f", d}')
      echo "${backend/:/_},$res,$rate,$achieved,$sent,$received,$drop,$(pct "$tmp/lat.csv" 7 0.5 0)" | tee -a "$out/node_rate.csv"
    done
  done
done
