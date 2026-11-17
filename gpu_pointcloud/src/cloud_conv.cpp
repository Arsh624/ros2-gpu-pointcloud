#include "gpu_pointcloud/cloud_conv.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include <sensor_msgs/msg/point_field.hpp>

namespace gpu_pointcloud {
namespace {

using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;

bool host_is_big_endian() {
  const std::uint16_t probe = 1;
  std::uint8_t first;
  std::memcpy(&first, &probe, 1);
  return first == 0;
}

// Byte offset of a float32 field, validated to lie inside one point.
std::size_t float32_offset(const PointCloud2& msg, const char* name) {
  for (const PointField& f : msg.fields) {
    if (f.name != name) continue;
    if (f.datatype != PointField::FLOAT32) {
      throw std::invalid_argument(std::string("PointCloud2 field '") + name + "' is not float32");
    }
    if (static_cast<std::size_t>(f.offset) + sizeof(float) > msg.point_step) {
      throw std::invalid_argument(std::string("PointCloud2 field '") + name + "' extends past point_step");
    }
    return f.offset;
  }
  throw std::invalid_argument(std::string("PointCloud2 has no '") + name + "' field");
}

}  // namespace

void from_pointcloud2(const PointCloud2& msg, std::vector<PointXYZ>& out) {
  out.clear();
  if (msg.is_bigendian != host_is_big_endian()) {
    throw std::invalid_argument("PointCloud2 endianness differs from the host; not supported");
  }
  const std::size_t ox = float32_offset(msg, "x");
  const std::size_t oy = float32_offset(msg, "y");
  const std::size_t oz = float32_offset(msg, "z");

  const std::size_t width = msg.width, height = msg.height;
  const std::size_t point_step = msg.point_step, row_step = msg.row_step;
  if (width == 0 || height == 0) return;
  if (row_step < width * point_step) {
    throw std::invalid_argument("PointCloud2 row_step is smaller than width * point_step");
  }
  if (msg.data.size() < (height - 1) * row_step + width * point_step) {
    throw std::invalid_argument("PointCloud2 data is shorter than its width, height and steps imply");
  }

  out.resize(width * height);
  PointXYZ* dst = out.data();
  for (std::size_t r = 0; r < height; ++r) {
    const std::uint8_t* src = msg.data.data() + r * row_step;
    for (std::size_t c = 0; c < width; ++c, src += point_step, ++dst) {
      std::memcpy(&dst->x, src + ox, sizeof(float));
      std::memcpy(&dst->y, src + oy, sizeof(float));
      std::memcpy(&dst->z, src + oz, sizeof(float));
    }
  }
}

bool pointcloud2_view(const PointCloud2&, CloudView&) { return false; }  // tests first

void to_pointcloud2(const PointXYZ* points, std::size_t n, const std_msgs::msg::Header& header, PointCloud2& out) {
  out.header = header;
  out.height = 1;
  out.width = static_cast<std::uint32_t>(n);
  out.fields.resize(3);
  const char* names[3] = {"x", "y", "z"};
  for (std::uint32_t i = 0; i < 3; ++i) {
    out.fields[i].name = names[i];
    out.fields[i].offset = 4 * i;
    out.fields[i].datatype = PointField::FLOAT32;
    out.fields[i].count = 1;
  }
  out.is_bigendian = host_is_big_endian();
  out.point_step = sizeof(PointXYZ);
  out.row_step = static_cast<std::uint32_t>(n * sizeof(PointXYZ));
  out.is_dense = true;
  out.data.resize(n * sizeof(PointXYZ));
  if (n > 0) std::memcpy(out.data.data(), points, n * sizeof(PointXYZ));
}

}  // namespace gpu_pointcloud
