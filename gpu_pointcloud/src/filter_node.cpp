// pointcloud_filter_node: PointCloud2 in on `input`, range + voxel filtered PointCloud2 out on
// `output`. The backend (cpu, cpu_omp or cuda) is a parameter; the node code is the same for all.
//
// Optional latency log: with parameter `latency_csv` set, the node records one row per message
// and writes the file when it shuts down (nothing is written while it runs). Columns:
//   transport_ms  callback entry minus header.stamp (meaningful when publisher and node share a clock)
//   view_ms       locating x, y, z in the message (or copying them, for clouds with row padding)
//   filter_ms     Filter::apply
//   publish_ms    building the output message and publish(), which serialises it
//   total_ms      callback entry to return of publish()
//   cpu_ms        CPU time the whole process used during the callback (getrusage user + system,
//                 all threads); cpu_ms / total_ms is the average number of cores in use
#include <sys/resource.h>

#include <chrono>
#include <cstdio>
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
using Clock = std::chrono::steady_clock;

// Device buffers are preallocated for a 1280x720 cloud; larger inputs grow them once.
constexpr std::size_t kCudaReservePoints = 1280 * 720;

// User + system CPU time of this process so far, all threads, in milliseconds.
double process_cpu_ms() {
  rusage u{};
  getrusage(RUSAGE_SELF, &u);
  return (u.ru_utime.tv_sec + u.ru_stime.tv_sec) * 1e3 + (u.ru_utime.tv_usec + u.ru_stime.tv_usec) * 1e-3;
}

double ms_between(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

class FilterNode : public rclcpp::Node {
 public:
  FilterNode() : rclcpp::Node("pointcloud_filter_node") {
    gpu_pointcloud::FilterParams params;
    params.voxel_size = static_cast<float>(declare_parameter("voxel_size", 0.05));
    params.min_range = static_cast<float>(declare_parameter("min_range", 0.3));
    params.max_range = static_cast<float>(declare_parameter("max_range", 10.0));
    const std::string backend = declare_parameter("backend", std::string("cpu"));
    latency_csv_ = declare_parameter("latency_csv", std::string(""));
    // Threads for backend cpu_omp; 0 = all the CPUs OpenMP sees. Inside a node, next to the
    // middleware and other processes, fewer than all is usually faster (results/node_latency.csv).
    const int cpu_threads = static_cast<int>(declare_parameter("cpu_threads", 0));

    if (backend == "cpu") {
      filter_ = gpu_pointcloud::make_cpu_filter(params);
    } else if (backend == "cpu_omp") {
      filter_ = gpu_pointcloud::make_cpu_omp_filter(params, cpu_threads);
    } else if (backend == "cuda") {
      filter_ = gpu_pointcloud::make_cuda_filter(params, kCudaReservePoints);
    } else {
      throw std::invalid_argument("parameter 'backend' must be 'cpu', 'cpu_omp' or 'cuda', got '" + backend + "'");
    }
    if (!latency_csv_.empty()) rows_.reserve(1 << 16);

    pub_ = create_publisher<PointCloud2>("output", rclcpp::SensorDataQoS());
    sub_ = create_subscription<PointCloud2>(
        "input", rclcpp::SensorDataQoS(), [this](PointCloud2::ConstSharedPtr msg) { on_cloud(*msg); });
    RCLCPP_INFO(get_logger(), "backend=%s voxel_size=%.3f min_range=%.2f max_range=%.2f", backend.c_str(),
                params.voxel_size, params.min_range, params.max_range);
  }

  ~FilterNode() override {
    if (latency_csv_.empty()) return;
    std::FILE* f = std::fopen(latency_csv_.c_str(), "w");
    if (!f) {
      RCLCPP_ERROR(get_logger(), "cannot write %s", latency_csv_.c_str());
      return;
    }
    std::fprintf(f, "points_in,points_out,transport_ms,view_ms,filter_ms,publish_ms,total_ms,cpu_ms\n");
    for (const Row& r : rows_) {
      std::fprintf(f, "%zu,%zu,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", r.points_in, r.points_out, r.transport_ms, r.view_ms,
                   r.filter_ms, r.publish_ms, r.total_ms, r.cpu_ms);
    }
    std::fclose(f);
  }

 private:
  struct Row {
    std::size_t points_in, points_out;
    double transport_ms, view_ms, filter_ms, publish_ms, total_ms, cpu_ms;
  };

  void on_cloud(const PointCloud2& msg) {
    const double cpu0 = latency_csv_.empty() ? 0.0 : process_cpu_ms();
    const auto t0 = Clock::now();
    const double transport_ms = (now() - rclcpp::Time(msg.header.stamp)).seconds() * 1e3;

    // Read the points where they are when the layout allows it; copy only for padded rows.
    gpu_pointcloud::CloudView view;
    try {
      if (!gpu_pointcloud::pointcloud2_view(msg, view)) {
        gpu_pointcloud::from_pointcloud2(msg, copied_);
        view = gpu_pointcloud::as_view(copied_.data(), copied_.size());
      }
    } catch (const std::invalid_argument& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "dropping cloud: %s", e.what());
      return;
    }
    const auto t1 = Clock::now();
    const gpu_pointcloud::FilterStats stats = filter_->apply(view, out_);
    const auto t2 = Clock::now();
    auto out_msg = std::make_unique<PointCloud2>();
    gpu_pointcloud::to_pointcloud2(out_.data(), out_.size(), msg.header, *out_msg);
    pub_->publish(std::move(out_msg));
    const auto t3 = Clock::now();

    if (!latency_csv_.empty()) {
      rows_.push_back({stats.in, stats.out, transport_ms, ms_between(t0, t1), ms_between(t1, t2), ms_between(t2, t3),
                       ms_between(t0, t3), process_cpu_ms() - cpu0});
    }
    RCLCPP_DEBUG(get_logger(), "in=%zu out=%zu non_finite=%zu out_of_range=%zu index_overflow=%zu", stats.in,
                 stats.out, stats.non_finite, stats.out_of_range, stats.index_overflow);
  }

  std::unique_ptr<gpu_pointcloud::Filter> filter_;
  std::vector<gpu_pointcloud::PointXYZ> copied_, out_;  // reused across frames
  rclcpp::Publisher<PointCloud2>::SharedPtr pub_;
  rclcpp::Subscription<PointCloud2>::SharedPtr sub_;
  std::string latency_csv_;
  std::vector<Row> rows_;
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
