// cloud_publisher_node: publishes the deterministic synthetic cloud on `cloud` at a fixed rate,
// stamped with the current time. Used to exercise the filter node and for the network benchmark.
#include <chrono>
#include <memory>

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

  const auto cloud = gpu_pointcloud::make_synthetic_cloud(width, height);
  sensor_msgs::msg::PointCloud2 msg;
  std_msgs::msg::Header header;
  header.frame_id = "camera";
  gpu_pointcloud::to_pointcloud2(cloud.data(), cloud.size(), header, msg);
  msg.is_dense = false;  // the raw synthetic cloud contains NaN pixels

  auto pub = node->create_publisher<sensor_msgs::msg::PointCloud2>("cloud", rclcpp::SensorDataQoS());
  std::int64_t sent = 0;
  auto timer = node->create_wall_timer(std::chrono::duration<double>(1.0 / rate_hz), [&]() {
    msg.header.stamp = node->now();
    pub->publish(msg);
    if (count > 0 && ++sent >= count) rclcpp::shutdown();
  });
  RCLCPP_INFO(node->get_logger(), "publishing %dx%d (%zu points) at %.1f Hz", width, height, cloud.size(), rate_hz);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
