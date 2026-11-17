// cloud_publisher_node: publishes the deterministic synthetic cloud on `cloud` at a fixed rate,
// stamped with the current time. Used to exercise the filter node and for the network benchmark.
//
// Parameters: width, height, rate (Hz), count (0 = until shut down), point_step (bytes per point,
// 12 = packed xyz; larger values pad each point like a real sensor driver's xyz + extra fields),
// wait_for_subscribers (do not start until this many subscribers are matched).
// On exit it logs how many messages it sent and the rate it actually achieved.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "gpu_pointcloud/cloud_conv.hpp"
#include "gpu_pointcloud/synthetic.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("cloud_publisher_node");
  const int width = static_cast<int>(node->declare_parameter("width", 640));
  const int height = static_cast<int>(node->declare_parameter("height", 480));
  const double rate_hz = node->declare_parameter("rate", 10.0);
  const std::int64_t count = node->declare_parameter("count", 0);  // 0 = publish until shut down
  const std::int64_t point_step = node->declare_parameter("point_step", 12);
  const std::int64_t wait_for = node->declare_parameter("wait_for_subscribers", 0);
  if (point_step < 12) {
    RCLCPP_FATAL(node->get_logger(), "point_step must be at least 12");
    return 1;
  }

  const auto cloud = gpu_pointcloud::make_synthetic_cloud(width, height);
  sensor_msgs::msg::PointCloud2 msg;
  std_msgs::msg::Header header;
  header.frame_id = "camera";
  gpu_pointcloud::to_pointcloud2(cloud.data(), cloud.size(), header, msg);
  msg.is_dense = false;  // the raw synthetic cloud contains NaN pixels
  if (point_step != 12) {
    // Same x, y, z fields at offsets 0, 4, 8, with filler bytes after them.
    msg.point_step = static_cast<std::uint32_t>(point_step);
    msg.row_step = msg.point_step * msg.width;
    msg.data.assign(cloud.size() * point_step, 0);
    for (std::size_t i = 0; i < cloud.size(); ++i) {
      std::memcpy(&msg.data[i * point_step], &cloud[i], sizeof(gpu_pointcloud::PointXYZ));
    }
  }

  auto pub = node->create_publisher<sensor_msgs::msg::PointCloud2>("cloud", rclcpp::SensorDataQoS());
  // Best-effort messages published before a subscriber is matched are simply lost, which would
  // look like drops in a benchmark. Optionally wait for the expected subscribers first.
  for (int waited_ms = 0; rclcpp::ok() && static_cast<std::int64_t>(pub->get_subscription_count()) < wait_for &&
                          waited_ms < 20000; waited_ms += 50) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  if (wait_for > 0) std::this_thread::sleep_for(std::chrono::milliseconds(300));

  std::int64_t sent = 0;
  const auto start = std::chrono::steady_clock::now();
  auto last = start;
  auto timer = node->create_wall_timer(std::chrono::duration<double>(1.0 / rate_hz), [&]() {
    msg.header.stamp = node->now();
    pub->publish(msg);
    last = std::chrono::steady_clock::now();
    if (++sent >= count && count > 0) rclcpp::shutdown();
  });
  RCLCPP_INFO(node->get_logger(), "publishing %dx%d (%zu points, %zu bytes) at %.1f Hz", width, height, cloud.size(),
              msg.data.size(), rate_hz);
  rclcpp::spin(node);
  const double seconds = std::chrono::duration<double>(last - start).count();
  // printf, not RCLCPP_INFO: the context may already be shut down here.
  std::printf("publisher_summary sent=%lld achieved_hz=%.2f requested_hz=%.2f\n", static_cast<long long>(sent),
              seconds > 0.0 ? sent / seconds : 0.0, rate_hz);
  rclcpp::shutdown();
  return 0;
}
