#include <gtest/gtest.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_pointcloud/cloud_conv.hpp"

using gpu_pointcloud::PointXYZ;
using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;

namespace {

PointField field(const std::string& name, std::uint32_t offset, std::uint8_t datatype = PointField::FLOAT32) {
  PointField f;
  f.name = name;
  f.offset = offset;
  f.datatype = datatype;
  f.count = 1;
  return f;
}

// A cloud like a real RGB-D driver publishes: x, y, z, 4 bytes padding, rgb, padding; point_step 32.
PointCloud2 make_xyzrgb(std::uint32_t width, std::uint32_t height, std::vector<PointXYZ>* truth = nullptr) {
  PointCloud2 msg;
  msg.header.frame_id = "camera";
  msg.header.stamp.sec = 12;
  msg.header.stamp.nanosec = 34;
  msg.width = width;
  msg.height = height;
  msg.fields = {field("x", 0), field("y", 4), field("z", 8), field("rgb", 16)};
  msg.point_step = 32;
  msg.row_step = msg.point_step * width;
  msg.is_bigendian = false;
  msg.is_dense = false;
  msg.data.assign(static_cast<std::size_t>(msg.row_step) * height, 0xAB);  // junk in the padding
  for (std::uint32_t i = 0; i < width * height; ++i) {
    const PointXYZ p{0.5f * i, -1.f - i, 100.f + i};
    std::memcpy(&msg.data[i * msg.point_step], &p, sizeof(p));
    if (truth) truth->push_back(p);
  }
  return msg;
}

void expect_same(const std::vector<PointXYZ>& a, const std::vector<PointXYZ>& b) {
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i].x, b[i].x) << i;
    EXPECT_EQ(a[i].y, b[i].y) << i;
    EXPECT_EQ(a[i].z, b[i].z) << i;
  }
}

}  // namespace

TEST(CloudConv, ReadsXyzFromCloudWithRgbAndPointStep32) {
  std::vector<PointXYZ> truth;
  const PointCloud2 msg = make_xyzrgb(5, 1, &truth);
  std::vector<PointXYZ> got{{9.f, 9.f, 9.f}};
  gpu_pointcloud::from_pointcloud2(msg, got);
  expect_same(got, truth);
}

TEST(CloudConv, RoundTripsThroughDenseXyzOutput) {
  std::vector<PointXYZ> truth;
  const PointCloud2 in = make_xyzrgb(5, 1, &truth);
  std::vector<PointXYZ> pts;
  gpu_pointcloud::from_pointcloud2(in, pts);

  PointCloud2 out;
  gpu_pointcloud::to_pointcloud2(pts.data(), pts.size(), in.header, out);
  EXPECT_EQ(out.header.frame_id, "camera");
  EXPECT_EQ(out.header.stamp.sec, 12);
  EXPECT_EQ(out.header.stamp.nanosec, 34u);
  EXPECT_EQ(out.height, 1u);
  EXPECT_EQ(out.width, 5u);
  EXPECT_EQ(out.point_step, 12u);
  EXPECT_EQ(out.row_step, 60u);
  EXPECT_EQ(out.data.size(), 60u);
  EXPECT_TRUE(out.is_dense);
  EXPECT_FALSE(out.is_bigendian);
  ASSERT_EQ(out.fields.size(), 3u);
  EXPECT_EQ(out.fields[0].name, "x");
  EXPECT_EQ(out.fields[2].name, "z");
  EXPECT_EQ(out.fields[2].offset, 8u);
  EXPECT_EQ(out.fields[1].datatype, PointField::FLOAT32);

  std::vector<PointXYZ> back;
  gpu_pointcloud::from_pointcloud2(out, back);
  expect_same(back, truth);
}

TEST(CloudConv, HandlesOrganizedCloudWithRowPaddingAndShiftedFields) {
  // 3x2 organized cloud, an intensity field first so x/y/z sit at offsets 4/8/12, and each row
  // padded by 8 bytes beyond width * point_step.
  PointCloud2 msg;
  msg.width = 3;
  msg.height = 2;
  msg.fields = {field("intensity", 0), field("x", 4), field("y", 8), field("z", 12)};
  msg.point_step = 16;
  msg.row_step = 3 * 16 + 8;
  msg.data.assign(msg.row_step * msg.height, 0);
  std::vector<PointXYZ> truth;
  for (std::uint32_t r = 0; r < 2; ++r) {
    for (std::uint32_t c = 0; c < 3; ++c) {
      const PointXYZ p{1.f * c, 10.f * r, -1.f};
      std::memcpy(&msg.data[r * msg.row_step + c * msg.point_step + 4], &p, sizeof(p));
      truth.push_back(p);
    }
  }
  std::vector<PointXYZ> got;
  gpu_pointcloud::from_pointcloud2(msg, got);
  expect_same(got, truth);
}

TEST(CloudConv, EmptyCloudGivesNoPoints) {
  PointCloud2 msg = make_xyzrgb(0, 1);
  std::vector<PointXYZ> got{{9.f, 9.f, 9.f}};
  gpu_pointcloud::from_pointcloud2(msg, got);
  EXPECT_TRUE(got.empty());

  PointCloud2 out;
  gpu_pointcloud::to_pointcloud2(nullptr, 0, msg.header, out);
  EXPECT_EQ(out.width, 0u);
  EXPECT_TRUE(out.data.empty());
  EXPECT_EQ(out.fields.size(), 3u);
}

TEST(CloudConv, RejectsCloudsItCannotReadSafely) {
  std::vector<PointXYZ> got;

  PointCloud2 no_z = make_xyzrgb(5, 1);
  no_z.fields.erase(no_z.fields.begin() + 2);
  EXPECT_THROW(gpu_pointcloud::from_pointcloud2(no_z, got), std::invalid_argument);

  PointCloud2 double_x = make_xyzrgb(5, 1);
  double_x.fields[0].datatype = PointField::FLOAT64;
  EXPECT_THROW(gpu_pointcloud::from_pointcloud2(double_x, got), std::invalid_argument);

  PointCloud2 short_data = make_xyzrgb(5, 1);
  short_data.data.resize(short_data.data.size() - 1);
  EXPECT_THROW(gpu_pointcloud::from_pointcloud2(short_data, got), std::invalid_argument);

  PointCloud2 field_past_step = make_xyzrgb(5, 1);
  field_past_step.fields[2].offset = 30;  // z would read 2 bytes into the next point
  EXPECT_THROW(gpu_pointcloud::from_pointcloud2(field_past_step, got), std::invalid_argument);

  PointCloud2 narrow_rows = make_xyzrgb(5, 1);
  narrow_rows.row_step = 100;  // less than width * point_step
  EXPECT_THROW(gpu_pointcloud::from_pointcloud2(narrow_rows, got), std::invalid_argument);

  PointCloud2 big_endian = make_xyzrgb(5, 1);
  big_endian.is_bigendian = true;
  EXPECT_THROW(gpu_pointcloud::from_pointcloud2(big_endian, got), std::invalid_argument);
}
