#!/usr/bin/env bash
# Network benchmark: Fast DDS against rmw_zenoh, publisher and listener in two containers on one
# Docker bridge network (docker/compose.net.yml). Runs on the host, from the repository root:
#   bash scripts/net_bench.sh
# Per RMW, payload (raw 640x480 cloud, or the same cloud after the default filter) and rate
# (10, 30 Hz): start the listener, publish SAMPLES + SKIP messages, then record what arrived.
# Latency is now - header.stamp at the listener; both containers share the host's clock.
# Writes results/net.csv. Per-message logs go to build_tmp/net/ (not committed).
#
# Environment: SAMPLES (2000), SKIP (20, start-up messages left out of the statistics),
# RMWS (default: "rmw_fastrtps_cpp rmw_zenoh_cpp"), PAYLOADS ("raw filtered"), RATES ("10 30"),
# OUT (results/net.csv). An RMW written rmw_fastrtps_cpp+LARGE_DATA runs Fast DDS with
# FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA (the one tuned configuration, labelled as such).
set -euo pipefail
export MSYS_NO_PATHCONV=1   # Git Bash on Windows: keep container paths as they are

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
samples="${SAMPLES:-2000}"
skip="${SKIP:-20}"
rmws="${RMWS:-rmw_fastrtps_cpp rmw_zenoh_cpp}"
payloads="${PAYLOADS:-raw filtered}"
rates="${RATES:-10 30}"
out="${OUT:-results/net.csv}"
logs="build_tmp/net"
mkdir -p "$logs" "$(dirname "$out")"

dc() { docker compose -f docker/compose.net.yml "$@"; }
dc up -d >/dev/null
# The talker builds the package into the install volume that the listener shares.
dc exec -T talker /ros_entrypoint.sh bash -c \
  'cd /ws && colcon build --cmake-args -DWITH_CUDA=OFF -DBUILD_TESTING=OFF > /tmp/build.log 2>&1 || { tail -20 /tmp/build.log; exit 1; }'

# field NAME FILE: value of NAME=... on the summary line in FILE ("nan" when missing).
field() { grep -o "$1=[^ ]*" "$2" | tail -1 | cut -d= -f2 || true; }

echo "rmw,payload,rate_hz,bytes,sent,received,drop_pct,p50_ms,p99_ms,max_ms,msgs_per_s,mb_per_s" > "$out"
for rmw_label in $rmws; do
  rmw="${rmw_label%%+*}"
  env=(-e "RMW_IMPLEMENTATION=$rmw")
  if [ "$rmw" = rmw_zenoh_cpp ]; then
    # Connect to the router; listen on all interfaces, not loopback, so peers can reach each other.
    env+=(-e 'ZENOH_CONFIG_OVERRIDE=connect/endpoints=["tcp/router:7447"];listen/endpoints=["tcp/0.0.0.0:0"]')
  fi
  [ "$rmw_label" = "rmw_fastrtps_cpp+LARGE_DATA" ] && env+=(-e FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA)
  for payload in $payloads; do
    filter=false
    [ "$payload" = filtered ] && filter=true
    for rate in $rates; do
      tag="${rmw_label}_${payload}_${rate}hz"
      dc exec -T "${env[@]}" listener /ros_entrypoint.sh ros2 run gpu_pointcloud latency_listener_node \
        --ros-args -p samples:="$samples" -p skip:="$skip" -p csv:="/ws/src/ros2-gpu-pointcloud/$logs/$tag.csv" \
        > "$logs/$tag.listener.log" 2>&1 &
      listener=$!
      dc exec -T "${env[@]}" talker /ros_entrypoint.sh ros2 run gpu_pointcloud cloud_publisher_node \
        --ros-args -p rate:="$rate.0" -p count:=$((samples + skip)) -p wait_for_subscribers:=1 -p filter:="$filter" \
        > "$logs/$tag.publisher.log" 2>&1
      wait "$listener" || true
      sent=$(field sent "$logs/$tag.publisher.log")
      bytes=$(grep -o '[0-9]* bytes)' "$logs/$tag.publisher.log" | head -1 | cut -d' ' -f1)
      received=$(field received "$logs/$tag.listener.log")
      received="${received:-0}"
      drop=$(awk -v s="${sent:-0}" -v r="$received" 'BEGIN {if (s > 0) printf "%.1f", 100 * (s - r) / s; else print "nan"}')
      echo "$rmw_label,$payload,$rate,$bytes,${sent:-0},$received,$drop,$(field p50_ms "$logs/$tag.listener.log"),$(field p99_ms "$logs/$tag.listener.log"),$(field max_ms "$logs/$tag.listener.log"),$(field msgs_per_s "$logs/$tag.listener.log"),$(field mb_per_s "$logs/$tag.listener.log")" |
        tee -a "$out"
    done
  done
done
