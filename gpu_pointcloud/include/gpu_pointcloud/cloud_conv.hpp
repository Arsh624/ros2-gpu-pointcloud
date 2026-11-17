// sensor_msgs/PointCloud2 <-> PointXYZ conversion. The only core-adjacent header that needs ROS.
#pragma once

#include <cstddef>
#include <vector>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>

#include "gpu_pointcloud/filter.hpp"

namespace gpu_pointcloud {

// Extracts x, y, z from any little-endian PointCloud2 that has float32 fields named "x", "y" and
// "z", whatever its point_step, field offsets, extra fields or row padding. `out` is replaced.
// Throws std::invalid_argument when a field is missing or not float32, when the layout would
// read outside `data`, or when the cloud is big-endian.
void from_pointcloud2(const sensor_msgs::msg::PointCloud2& msg, std::vector<PointXYZ>& out);

// Writes an unorganized (height 1), dense, little-endian cloud with exactly the float32 fields
// x, y, z (point_step 12) and the given header.
void to_pointcloud2(const PointXYZ* points, std::size_t n, const std_msgs::msg::Header& header,
                    sensor_msgs::msg::PointCloud2& out);

}  // namespace gpu_pointcloud
