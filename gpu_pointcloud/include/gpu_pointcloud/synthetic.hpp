// Deterministic synthetic depth-camera point cloud. ROS-free.
#pragma once

#include <cstdint>
#include <vector>

#include "gpu_pointcloud/filter.hpp"

namespace gpu_pointcloud {

// Simulates a pinhole depth camera at the origin looking along +z (x right, y down), with
// fx = fy = 0.8 * width and the principal point at the image centre. The scene is a floor, a
// back wall that is partly beyond 10 m, three boxes and two spheres.
//
// Returns exactly width * height points in row-major pixel order. About 5% of them are NaN
// (invalid pixels) and depth carries Gaussian noise with sigma 5 mm. The same arguments always
// give the same cloud when built with the same standard library.
std::vector<PointXYZ> make_synthetic_cloud(int width, int height, std::uint32_t seed = 42);

}  // namespace gpu_pointcloud
