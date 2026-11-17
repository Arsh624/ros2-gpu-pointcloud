# Architecture

One ament_cmake package, `gpu_pointcloud`, in two layers.

## Core library (`gpu_pointcloud_core`, no ROS)

| File | Role |
|---|---|
| `include/gpu_pointcloud/filter.hpp` | Point type, parameters, the `Filter` interface and backend factories |
| `src/filter_cpu.cpp` | Single-threaded reference backend |
| `src/filter_cuda.cu` | CUDA backend, built only with `WITH_CUDA=ON` |
| `include/gpu_pointcloud/detail/voxel_math.hpp` | Per-point classification (finite, range, voxel key) compiled for both host and device |
| `include/gpu_pointcloud/synthetic.hpp`, `src/synthetic.cpp` | Deterministic synthetic depth-camera cloud |

Tests and the benchmark link only this library, so they run without a ROS graph.

## ROS layer

| File | Role |
|---|---|
| `include/gpu_pointcloud/cloud_conv.hpp`, `src/cloud_conv.cpp` | `sensor_msgs/PointCloud2` to and from `PointXYZ` |
| `src/filter_node.cpp` | `pointcloud_filter_node`: subscribes `input`, publishes `output`, parameter `backend: cpu|cuda` |
| `src/cloud_publisher_node.cpp` | `cloud_publisher_node`: publishes the synthetic cloud on `cloud` at a fixed rate |

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
