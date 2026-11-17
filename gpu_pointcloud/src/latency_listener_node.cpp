// latency_listener_node: receives PointCloud2 on `cloud` (SensorDataQoS) and records, per message,
// latency = now - header.stamp and the payload size. The first `skip` messages (start-up) are
// counted as received but left out of the statistics. Publisher and listener must share a clock
// (same host). After `skip` + `samples` messages, or `idle_timeout_s` without a message once the first one
// arrived (`first_timeout_s` before it), it writes the CSV, prints one summary line and exits.
//
// Parameters: samples (2000), skip (0), csv (path, empty = no file), idle_timeout_s (10), first_timeout_s (60).
// Summary line: listener_summary received=N samples=S bytes=B p50_ms=.. p99_ms=.. max_ms=.. msgs_per_s=.. mb_per_s=..
// Drops are computed by the caller as sent - received, from the publisher's own summary line.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "gpu_pointcloud/latency_stats.hpp"

namespace {

double steady_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("latency_listener_node");
  const auto samples = static_cast<std::size_t>(node->declare_parameter("samples", 2000));
  const auto skip = static_cast<std::size_t>(node->declare_parameter("skip", 0));
  const std::string csv = node->declare_parameter("csv", std::string(""));
  const double idle_timeout_s = node->declare_parameter("idle_timeout_s", 10.0);
  const double first_timeout_s = node->declare_parameter("first_timeout_s", 60.0);

  std::vector<gpu_pointcloud::Arrival> arrivals;
  arrivals.reserve(samples + skip);
  const double start_s = steady_s();

  auto sub = node->create_subscription<sensor_msgs::msg::PointCloud2>(
      "cloud", rclcpp::SensorDataQoS(), [&](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
        const double latency_ms = (node->now() - rclcpp::Time(msg->header.stamp)).seconds() * 1e3;
        arrivals.push_back({steady_s(), latency_ms, msg->data.size()});
        if (arrivals.size() >= samples + skip) rclcpp::shutdown();
      });
  auto watchdog = node->create_wall_timer(std::chrono::milliseconds(200), [&]() {
    const double now_s = steady_s();
    if (arrivals.empty() ? now_s - start_s > first_timeout_s : now_s - arrivals.back().t_s > idle_timeout_s) {
      rclcpp::shutdown();
    }
  });
  RCLCPP_INFO(node->get_logger(), "waiting for %zu + %zu messages on %s", skip, samples, sub->get_topic_name());
  rclcpp::spin(node);

  if (!csv.empty()) {
    std::FILE* f = std::fopen(csv.c_str(), "w");
    if (!f) {
      std::fprintf(stderr, "error: cannot write %s\n", csv.c_str());
      return 1;
    }
    std::fprintf(f, "arrival_s,latency_ms,bytes\n");
    for (const auto& a : arrivals) {
      std::fprintf(f, "%.6f,%.4f,%llu\n", a.t_s - start_s, a.latency_ms, static_cast<unsigned long long>(a.bytes));
    }
    std::fclose(f);
  }
  const std::vector<gpu_pointcloud::Arrival> kept(arrivals.begin() + static_cast<std::ptrdiff_t>(std::min(skip, arrivals.size())),
                                                   arrivals.end());
  const auto s = gpu_pointcloud::summarize(kept);
  // printf, not RCLCPP_INFO: the context is already shut down here.
  std::printf("listener_summary received=%zu samples=%zu bytes=%llu p50_ms=%.3f p99_ms=%.3f max_ms=%.3f msgs_per_s=%.2f mb_per_s=%.2f\n",
              arrivals.size(), s.received, static_cast<unsigned long long>(s.bytes), s.p50_ms, s.p99_ms, s.max_ms, s.msgs_per_s,
              s.mb_per_s);
  return 0;
}
