# ros2-gpu-pointcloud

[![CI](https://github.com/Arsh624/ros2-gpu-pointcloud/actions/workflows/ci.yml/badge.svg)](https://github.com/Arsh624/ros2-gpu-pointcloud/actions/workflows/ci.yml)

A ROS 2 Jazzy node that range-filters and voxel-downsamples point clouds, with a single-threaded C++17 backend and a CUDA backend that produce the same output, plus a benchmark of the two and a DDS vs Zenoh transport comparison.

**Work in progress.** See [docs/STATUS.md](docs/STATUS.md) for what exists today.

CI builds and tests the CPU path and compiles the CUDA code. GitHub runners have no GPU, so the GPU tests run only locally.
