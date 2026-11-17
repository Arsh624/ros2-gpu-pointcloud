# Changelog

## Unreleased

- Repository skeleton: Docker image, `gpu_pointcloud` package, CI.
- CPU range + voxel filter, PointCloud2 conversion, `pointcloud_filter_node`, synthetic cloud publisher.
- CUDA filter with CPU parity tests.
- `bench_filter` benchmark with committed results.
- OpenMP CPU backend and PCL VoxelGrid baseline; benchmark reports speedup against the best CPU baseline.
- Benchmark: per-iteration samples, time-based warmup, and `bench_idle_gap` (call time after idle gaps). The CUDA p99 tail is explained (GPU power state after idle).
- All backends filter strided clouds in place (`CloudView`); the CUDA backend uploads the input buffer directly instead of staging it through pinned memory.
