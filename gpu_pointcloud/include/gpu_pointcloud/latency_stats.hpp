// Summary statistics for latency_listener_node. No ROS, so it is unit-tested on its own.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace gpu_pointcloud {

// One received message: arrival time (any monotonic clock, seconds), latency (ms), payload bytes.
struct Arrival {
  double t_s;
  double latency_ms;
  std::uint64_t bytes;
};

struct LatencySummary {
  std::size_t received = 0;
  std::uint64_t bytes = 0;  // payload size of the last message
  double p50_ms = std::numeric_limits<double>::quiet_NaN();
  double p99_ms = std::numeric_limits<double>::quiet_NaN();
  double max_ms = std::numeric_limits<double>::quiet_NaN();
  double msgs_per_s = 0.0;  // messages after the first, over first-to-last arrival
  double mb_per_s = 0.0;    // payload bytes of those messages, 1 MB = 1e6 bytes
};

// Nearest-rank percentile, p in (0, 1]: the value at 1-based index ceil(p * n). NaN when empty.
inline double percentile(std::vector<double> values, double p) {
  if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
  std::sort(values.begin(), values.end());
  const auto n = values.size();
  auto rank = static_cast<std::size_t>(std::ceil(p * static_cast<double>(n) - 1e-9));
  rank = std::min(std::max<std::size_t>(rank, 1), n);
  return values[rank - 1];
}

inline LatencySummary summarize(const std::vector<Arrival>& arrivals) {
  LatencySummary s;
  s.received = arrivals.size();
  if (arrivals.empty()) return s;
  std::vector<double> lat;
  lat.reserve(arrivals.size());
  for (const Arrival& a : arrivals) lat.push_back(a.latency_ms);
  s.p50_ms = percentile(lat, 0.50);
  s.p99_ms = percentile(lat, 0.99);
  s.max_ms = percentile(lat, 1.00);
  s.bytes = arrivals.back().bytes;
  const double window_s = arrivals.back().t_s - arrivals.front().t_s;
  if (arrivals.size() > 1 && window_s > 0.0) {
    double bytes_after_first = 0.0;
    for (std::size_t i = 1; i < arrivals.size(); ++i) bytes_after_first += static_cast<double>(arrivals[i].bytes);
    s.msgs_per_s = static_cast<double>(arrivals.size() - 1) / window_s;
    s.mb_per_s = bytes_after_first / 1e6 / window_s;
  }
  return s;
}

}  // namespace gpu_pointcloud
