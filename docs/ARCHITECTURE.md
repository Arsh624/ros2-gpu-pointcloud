# Architecture

One ament_cmake package, `gpu_pointcloud`, in two layers.

## Core library (`gpu_pointcloud_core`, no ROS)

| File | Role |
|---|---|
| `include/gpu_pointcloud/filter.hpp` | Point type, `CloudView` (strided input), parameters, the `Filter` interface and backend factories |
| `src/filter_cpu.cpp` | Single-threaded reference backend |
| `src/filter_cpu_omp.cpp` | OpenMP backend, bit-identical to the reference |
| `src/filter_cuda.cu` | CUDA backend, built only with `WITH_CUDA=ON` |
| `include/gpu_pointcloud/detail/voxel_math.hpp` | Per-point classification (finite, range, voxel key) compiled for both host and device |
| `include/gpu_pointcloud/synthetic.hpp`, `src/synthetic.cpp` | Deterministic synthetic depth-camera cloud |

Tests and the benchmark link only this library, so they run without a ROS graph.

## ROS layer

| File | Role |
|---|---|
| `include/gpu_pointcloud/cloud_conv.hpp`, `src/cloud_conv.cpp` | `sensor_msgs/PointCloud2` to and from `PointXYZ` |
| `src/filter_node.cpp` | `pointcloud_filter_node`: subscribes `input`, publishes `output`; parameters `backend: cpu|cpu_omp|cuda`, `cpu_threads`, `voxel_size`, `min_range`, `max_range`, `latency_csv` (per-message timing and CPU log) |
| `src/cloud_publisher_node.cpp` | `cloud_publisher_node`: publishes the synthetic cloud on `cloud` at a fixed rate; `point_step`, `count`, `wait_for_subscribers` |
| `docker/fastdds_shm_large.xml` | Fast DDS shared-memory profile with a 16 MB segment, set in the dev container (see DESIGN.md, "Middleware for same-host node runs") |



## Build switches

- `WITH_CUDA` (default ON when a CUDA compiler is found): compiles the CUDA backend. Without it, asking for the CUDA backend fails with a clear error.
- `CORE_ONLY` (default OFF): builds only the core library with plain CMake, no ament. CI uses it to compile the CUDA code in an image that has no ROS.

## Environments

- `docker/Dockerfile`: CUDA 12.6 devel on Ubuntu 24.04 plus ROS 2 Jazzy. The repository is mounted at `/ws/src/ros2-gpu-pointcloud` and built with colcon in `/ws`.
- CI: a CPU build and test job in `ros:jazzy-ros-base`, and a CUDA compile-only job.

## Tests

| File | Covers |
|---|---|
| `test/filter_cases.hpp` | Backend-independent filter cases (gtest value-parameterized), instantiated per backend |
| `test/test_filter_cpu.cpp` | The cases on the CPU backend, voxel key packing, synthetic scene determinism |
| `test/test_cloud_conv.cpp` | PointCloud2 conversion, including clouds it must reject |
| `test/test_parity_cuda.cpp` | The cases on the CUDA backend and CPU-vs-CUDA parity; skipped without a GPU |
| `test/test_pcl_baseline.cpp` | Checks the PCL baseline does comparable work (`WITH_PCL=ON` only) |

## Benchmarks and scripts

| File | Role |
|---|---|
| `bench/bench_filter.cpp` | Benchmark of every backend, no ROS; writes `results/bench.csv` and `results/stages.csv` |
| `bench/pcl_baseline.{hpp,cpp}` | PCL range + VoxelGrid reference pipeline for the benchmark (`WITH_PCL=ON` only) |
| `bench/bench_cuda_input.cpp` | Raw upload vs host repack, interleaved in one process; rows for `results/cuda_input_modes.csv` |
| `bench/bench_idle_gap.cpp` | Call time after controlled idle gaps and recovery after 3 s idle; writes `results/idle_gap.csv` and `results/idle_recovery.csv` |
| `bench/bench_h2d.cu` | Bare host-to-device transfer time by source-buffer allocation method |
| `scripts/bench.sh` | Runs the benchmarks under the 20 + 200 protocol and records the environment in `results/env.txt` |
| `scripts/profile.sh` | Nsight Systems capture and `results/nsys_summary.txt` |
| `scripts/node_bench.sh` | Runs the node against the publisher at 10 and 30 Hz per backend; writes `results/node_latency.csv` |
| `scripts/plot_before_after.py` | Plots the files from `bench_filter --dump` into `results/before_after.png` (host Python with matplotlib) |

`results/` holds the current run; `results/history/<name>/` holds earlier runs for before/after comparison.
