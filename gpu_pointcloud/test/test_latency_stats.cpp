// Summary arithmetic of latency_listener_node: nearest-rank percentiles and arrival rates.
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "gpu_pointcloud/latency_stats.hpp"

using gpu_pointcloud::Arrival;
using gpu_pointcloud::percentile;
using gpu_pointcloud::summarize;

TEST(Percentile, NearestRankOnOneToHundred) {
  std::vector<double> v;
  for (int i = 100; i >= 1; --i) v.push_back(i);  // unsorted on purpose
  EXPECT_DOUBLE_EQ(percentile(v, 0.50), 50.0);
  EXPECT_DOUBLE_EQ(percentile(v, 0.99), 99.0);
  EXPECT_DOUBLE_EQ(percentile(v, 1.00), 100.0);
}

TEST(Percentile, SmallSampleRoundsUp) {
  // Nearest rank: index ceil(p * n). For n = 3, p50 is the 2nd value and p99 the 3rd.
  EXPECT_DOUBLE_EQ(percentile({3.0, 1.0, 2.0}, 0.50), 2.0);
  EXPECT_DOUBLE_EQ(percentile({3.0, 1.0, 2.0}, 0.99), 3.0);
}

TEST(Percentile, EmptyIsNan) { EXPECT_TRUE(std::isnan(percentile({}, 0.5))); }

TEST(Summarize, EmptyHasNoRates) {
  const auto s = summarize({});
  EXPECT_EQ(s.received, 0u);
  EXPECT_TRUE(std::isnan(s.p50_ms));
  EXPECT_DOUBLE_EQ(s.msgs_per_s, 0.0);
  EXPECT_DOUBLE_EQ(s.mb_per_s, 0.0);
}

TEST(Summarize, RatesOverTheArrivalWindow) {
  // 11 messages of 1 MB arriving every 0.1 s: 10 intervals in 1 s -> 10 msg/s and 10 MB/s.
  std::vector<Arrival> a;
  for (int i = 0; i <= 10; ++i) a.push_back({0.1 * i, 2.0 + i, 1000000});
  const auto s = summarize(a);
  EXPECT_EQ(s.received, 11u);
  EXPECT_NEAR(s.msgs_per_s, 10.0, 1e-9);
  EXPECT_NEAR(s.mb_per_s, 10.0, 1e-9);
  EXPECT_DOUBLE_EQ(s.p50_ms, 7.0);   // latencies 2..12, 6th of 11
  EXPECT_DOUBLE_EQ(s.max_ms, 12.0);
  EXPECT_EQ(s.bytes, 1000000u);
}

TEST(Summarize, SingleMessageHasNoRate) {
  const auto s = summarize({{5.0, 3.5, 100}});
  EXPECT_EQ(s.received, 1u);
  EXPECT_DOUBLE_EQ(s.p50_ms, 3.5);
  EXPECT_DOUBLE_EQ(s.p99_ms, 3.5);
  EXPECT_DOUBLE_EQ(s.msgs_per_s, 0.0);
}
