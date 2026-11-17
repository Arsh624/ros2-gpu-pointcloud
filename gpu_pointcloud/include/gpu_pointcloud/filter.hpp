// Range + voxel-grid point cloud filter with CPU and CUDA backends.
// This header is ROS-free on purpose: tests and the benchmark use it without a ROS graph.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace gpu_pointcloud {

struct PointXYZ {
  float x, y, z;
};
static_assert(sizeof(PointXYZ) == 12, "PointXYZ must be 3 packed floats");

struct FilterParams {
  float voxel_size = 0.05f;  // metres
  float min_range = 0.3f;    // metres, inclusive
  float max_range = 10.0f;   // metres, inclusive
};

// Where every input point went. in == non_finite + out_of_range + index_overflow + (points kept).
struct FilterStats {
  std::size_t in = 0;
  std::size_t non_finite = 0;
  std::size_t out_of_range = 0;
  std::size_t index_overflow = 0;
  std::size_t out = 0;  // occupied voxels
};

// True when the library was compiled with the CUDA backend (WITH_CUDA=ON).
bool cuda_compiled();
// True when a usable CUDA device is visible at run time. Always false without cuda_compiled().
bool cuda_device_present();

}  // namespace gpu_pointcloud
