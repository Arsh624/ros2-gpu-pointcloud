# Changelog

## Unreleased

- Repository skeleton: Docker image, `gpu_pointcloud` package, CI.
- CPU range + voxel filter, PointCloud2 conversion, `pointcloud_filter_node`, synthetic cloud publisher.
- CUDA filter with CPU parity tests.
- `bench_filter` benchmark with committed results.
- OpenMP CPU backend and PCL VoxelGrid baseline; benchmark reports speedup against the best CPU baseline.
- All backends filter strided clouds in place (`CloudView`); the CUDA backend uploads the input buffer directly instead of staging it through pinned memory.
