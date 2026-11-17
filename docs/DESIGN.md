# Design notes

Decisions and the reasons for them. Interfaces are listed in [STATUS.md](STATUS.md).

## Filter semantics

Both backends apply the same five steps, in this order.

1. **Non-finite drop.** A point with any NaN or Inf coordinate is dropped.
2. **Range.** `r = sqrt(x*x + y*y + z*z)`, computed in double from the float32 coordinates. Keep the point when `min_range <= r <= max_range`, both ends inclusive.
3. **Voxel index.** Per axis `i = floor(coord / voxel_size)`, computed in float32. A point whose index falls outside `[-2^20, 2^20)` on any axis is dropped and counted in `FilterStats::index_overflow`. With the defaults that limit is about +/-52 km, so it only matters for a very small `voxel_size`.
4. **Key.** `key = (ix + 2^20) << 42 | (iy + 2^20) << 21 | (iz + 2^20)`, 21 bits per axis in a uint64.
5. **Centroid.** One output point per key: the mean of its points, summed in double, emitted as float32. Output is in ascending key order.

### Why these choices

- **Ascending key order** makes the output deterministic and lets the parity test compare the two backends element by element. No matching step is needed.
- **Float32 division for the index, double for the range.** IEEE 754 division, multiplication, addition and square root are correctly rounded on both x86 and the GPU, so the same inputs give the same bits provided the compiler does not fuse a multiply and an add into one FMA. Contraction is therefore switched off on both sides (`-ffp-contract=off` for GCC, `--fmad=false` for nvcc). This is what makes "same voxel count exactly" achievable.
- **Double accumulation.** Centroid sums are the only place where the two backends may add in a different order. In double the difference is many orders of magnitude below the 1e-4 m parity tolerance.
- **A point exactly on a voxel face** belongs to the voxel `floor` gives it: `x = 0.05` with `voxel_size = 0.05` has index 1, and `x = -0.0001` has index -1.

## Toolchain

Image: `nvidia/cuda:12.6.3-devel-ubuntu24.04@sha256:392c0df7b577ecae17a17f6ba7f2009c217bb4422f8431c053ae9af61a8c148a` plus the ROS 2 Jazzy apt repository.

Installed versions in the image built on 2026-11-16 (`dpkg -l`):

| Package | Version |
|---|---|
| cuda-nvcc-12-6 | 12.6.85-1 |
| cuda-cudart-dev-12-6 | 12.6.77-1 |
| cuda-sanitizer-12-6 | 12.6.77-1 |
| nsight-systems-2024.5.1 | 2024.5.1.113 |
| g++-13 | 13.2.0-23ubuntu4 |
| cmake | 3.28.3-1build7 |
| ros-jazzy-ros-base | 0.11.0-1noble.20261105.164630 |
| ros-jazzy-rclcpp | 28.1.23-1noble.20261105.084818 |
| ros-jazzy-sensor-msgs | 5.3.8-1noble.20261105.060814 |
| ros-jazzy-rmw-zenoh-cpp | 0.2.13-1noble.20261104.171211 |
| ros-jazzy-rmw-cyclonedds-cpp | 2.2.4-1noble.20261105.063358 |
| ros-jazzy-ament-cmake-gtest | 2.5.7-1noble.20261025.230436 |

The base image is pinned by digest. The ROS packages come from the rolling Jazzy apt repository and are not pinned, so a rebuild on a later date may pick up newer patch versions.

## Working in the dev container

`docker exec` does not run the image entrypoint, so ROS is not sourced by default. Run commands through it:

```
docker exec gpc-dev /ros_entrypoint.sh bash -c 'cd /ws && colcon build'
```

From Git Bash on Windows, set `MSYS_NO_PATHCONV=1` first so `/ros_entrypoint.sh` is not rewritten to a Windows path.
