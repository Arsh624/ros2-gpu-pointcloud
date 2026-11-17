# Changelog

## v0.2.0 - 2026-11-17

Networking: Fast DDS against rmw_zenoh between two containers on one Docker bridge network (`results/net.csv`).

### Added
- `latency_listener_node`: per-message latency (`now - header.stamp`) and payload size, CSV log and one summary line (p50, p99, max, msgs/s, MB/s); summary arithmetic in `latency_stats.hpp` with gtest cases.
- `cloud_publisher_node filter:=true`: publishes the filtered cloud (34,654 points, 415,848 bytes) as the second payload.
- `docker/compose.net.yml` (Zenoh router, talker, listener) and `scripts/net_bench.sh`.

### Results
- Raw 640x480 cloud (3,686,400 bytes), Fast DDS defaults: 37.4% dropped at 10 Hz, 32.5% at 30 Hz. Filtered cloud: 13.3% and 24.9%.
- Zenoh defaults: 0% dropped in every configuration; raw cloud p50 4.283 ms at 10 Hz and 4.002 ms at 30 Hz (110.59 MB/s).
- Fast DDS with `FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA` (the one tuned configuration): 0% dropped; raw cloud p50 4.154 and 4.136 ms.
- Single host, Docker bridge network, separate network namespaces (no shared memory), same clock at both ends; not a real LAN.

## v0.1.0 - 2026-11-17

First release: the filter, its three backends, the ROS 2 node and the benchmarks. Measured on an RTX 4070 Laptop GPU and an i9-14900HX under Docker on WSL2; every number below is in `results/`.

### Added
- Range filter and voxel-grid downsampling (centroid per voxel, ascending key order) with three backends that return the same voxel count and centroids within 1e-4 m: single-threaded C++17 (`cpu`), OpenMP (`cpu_omp`, bit-identical to `cpu`) and CUDA (`cuda`: classify kernel, `cub::DeviceSelect::Flagged`, `cub::DeviceRadixSort::SortPairs` on a 27-bit narrowed key, `cub::DeviceReduce::ReduceByKey`, buffers allocated once).
- `pointcloud_filter_node` (ROS 2 Jazzy): any PointCloud2 with float32 x, y, z and any `point_step`, read in place through a zero-copy view; parameters `backend`, `cpu_threads`, `voxel_size`, `min_range`, `max_range`, `latency_csv`. `cloud_publisher_node` publishes a deterministic synthetic depth-camera cloud.
- 81 tests with `-DWITH_PCL=ON` (gtest): CPU edge cases, PointCloud2 conversion, CPU/CUDA parity on 3 resolutions x 3 voxel sizes, PCL comparison. CI builds and tests the CPU paths and compiles the CUDA code.
- Benchmarks: `bench_filter` (all backends plus a PCL VoxelGrid baseline), `bench_idle_gap`, `bench_cuda_input`, `bench_h2d`, `scripts/bench.sh`, `scripts/node_bench.sh`, `scripts/profile.sh`, and `results/before_after.png`.

### Results
- Node at 1280x720 and 30 Hz: CUDA 3.48 ms per frame on 0.96 CPU cores; OpenMP 5.10 ms on 6.14 cores (20 threads) or 6.68 ms on 4.55 cores (12 threads). At 10 Hz CUDA takes 11.07 ms (`results/node_latency.csv`).
- Tight loop, CUDA end-to-end against OpenMP on 20 threads: 0.59x / 1.30x / 1.76x at 320x240 / 640x480 / 1280x720; GPU-only 1.13x / 3.57x / 6.64x (`results/bench.csv`).
- Input path: removing the pinned staging copy cut the 1280x720 input stage from 1.618 to 1.053 ms and CUDA end-to-end from 2.356 to 1.641 ms (`results/history/a-baselines`, `results/stages.csv`, `results/bench.csv`).
- Idle GPU: after 100 ms idle a 1280x720 CUDA call takes 6.96 ms against 1.49 ms in a tight loop; the GPU drops to P8 (210 MHz) when idle (`results/idle_gap.csv`, `results/idle_recovery.csv`). This also explained an earlier 11-12 ms CUDA p99.

### Fixed
- The image had no Fast DDS, so nodes ran on Cyclone DDS; it is installed now. Large best-effort clouds were mostly dropped in one container with middleware defaults; the dev container now uses Fast DDS shared memory with a 16 MB segment and a 1 GB `/dev/shm`.

### Known limitations
See the README's Limitations section: synthetic data, one laptop GPU under WSL2, compute-sanitizer not run, no GPU kernel rows in the Nsight capture, networking not measured yet.
