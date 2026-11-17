// pointcloud_filter_node: PointCloud2 in on `input`, range + voxel filtered PointCloud2 out on
// `output`. The backend (cpu or cuda) is a parameter; the node code is the same for both.
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "gpu_pointcloud/cloud_conv.hpp"
#include "gpu_pointcloud/filter.hpp"

namespace {

using sensor_msgs::msg::PointCloud2;

// Device buffers are preallocated for a 1280x720 cloud; larger inputs grow them once.
constexpr std::size_t kCudaReservePoints = 1280 * 720;

class FilterNode : public rclcpp::Node {
 public:
  FilterNode() : rclcpp::Node("pointcloud_filter_node") {
    gpu_pointcloud::FilterParams params;
    params.voxel_size = static_cast<float>(declare_parameter("voxel_size", 0.05));
    params.min_range = static_cast<float>(declare_parameter("min_range", 0.3));
    params.max_range = static_cast<float>(declare_parameter("max_range", 10.0));
    const std::string backend = declare_parameter("backend", std::string("cpu"));

    if (backend == "cpu") {
      filter_ = gpu_pointcloud::make_cpu_filter(params);
    } else if (backend == "cuda") {
      filter_ = gpu_pointcloud::make_cuda_filter(params, kCudaReservePoints);
    } else {
      throw std::invalid_argument("parameter 'backend' must be 'cpu' or 'cuda', got '" + backend + "'");
    }

    pub_ = create_publisher<PointCloud2>("output", rclcpp::SensorDataQoS());
    sub_ = create_subscription<PointCloud2>(
        "input", rclcpp::SensorDataQoS(), [this](PointCloud2::ConstSharedPtr msg) { on_cloud(*msg); });
    RCLCPP_INFO(get_logger(), "backend=%s voxel_size=%.3f min_range=%.2f max_range=%.2f", backend.c_str(),
                params.voxel_size, params.min_range, params.max_range);
  }

 private:
  void on_cloud(const PointCloud2& msg) {
    try {
      gpu_pointcloud::from_pointcloud2(msg, in_);
    } catch (const std::invalid_argument& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "dropping cloud: %s", e.what());
      return;
    }
    const gpu_pointcloud::FilterStats stats = filter_->apply(in_.data(), in_.size(), out_);
    auto out_msg = std::make_unique<PointCloud2>();
    gpu_pointcloud::to_pointcloud2(out_.data(), out_.size(), msg.header, *out_msg);
    pub_->publish(std::move(out_msg));
    RCLCPP_DEBUG(get_logger(), "in=%zu out=%zu non_finite=%zu out_of_range=%zu index_overflow=%zu", stats.in,
                 stats.out, stats.non_finite, stats.out_of_range, stats.index_overflow);
  }

  std::unique_ptr<gpu_pointcloud::Filter> filter_;
  std::vector<gpu_pointcloud::PointXYZ> in_, out_;  // reused across frames
  rclcpp::Publisher<PointCloud2>::SharedPtr pub_;
  rclcpp::Subscription<PointCloud2>::SharedPtr sub_;
};

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  int rc = 0;
  try {
    rclcpp::spin(std::make_shared<FilterNode>());
  } catch (const std::exception& e) {
    RCLCPP_FATAL(rclcpp::get_logger("pointcloud_filter_node"), "%s", e.what());
    rc = 1;
  }
  rclcpp::shutdown();
  return rc;
}
