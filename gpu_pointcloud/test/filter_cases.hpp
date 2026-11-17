// Backend-independent filter test cases. Each test executable instantiates the suite with the
// backends it wants (CPU in test_filter_cpu, CUDA in test_parity_cuda), so both backends are held
// to exactly the same expectations.
#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <ostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "gpu_pointcloud/filter.hpp"

namespace gpu_pointcloud::cases {

struct Backend {
  const char* name;
  bool needs_gpu;
  std::unique_ptr<Filter> (*make)(const FilterParams&);
};
inline std::ostream& operator<<(std::ostream& os, const Backend& b) { return os << b.name; }

class FilterCases : public ::testing::TestWithParam<Backend> {
 protected:
  void SetUp() override {
    if (GetParam().needs_gpu && !cuda_device_present()) {
      GTEST_SKIP() << "no CUDA device visible; GPU tests run only on a machine with a GPU";
    }
  }

  std::vector<PointXYZ> run(const std::vector<PointXYZ>& in, const FilterParams& params = {}) {
    auto filter = GetParam().make(params);
    std::vector<PointXYZ> out{{9.f, 9.f, 9.f}};  // stale content must be replaced, not appended to
    stats = filter->apply(in.data(), in.size(), out);
    EXPECT_EQ(stats.in, in.size());
    EXPECT_EQ(stats.out, out.size());
    return out;
  }

  FilterStats stats;
};

TEST_P(FilterCases, EmptyInputGivesEmptyOutput) {
  auto filter = GetParam().make({});
  std::vector<PointXYZ> out{{9.f, 9.f, 9.f}};
  const FilterStats s = filter->apply(nullptr, 0, out);
  EXPECT_TRUE(out.empty());
  EXPECT_EQ(s.in, 0u);
  EXPECT_EQ(s.out, 0u);
}

TEST_P(FilterCases, AllOutOfRangeGivesEmptyOutput) {
  const auto out = run({{0.1f, 0.f, 0.f}, {0.f, 0.f, 0.29f}, {20.f, 0.f, 0.f}, {6.f, 6.f, 6.f}});
  EXPECT_TRUE(out.empty());
  EXPECT_EQ(stats.out_of_range, 4u);
}

TEST_P(FilterCases, NonFiniteCoordinatesAreDropped) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  const auto out = run({{nan, 1.f, 1.f}, {1.f, inf, 1.f}, {1.f, 1.f, -inf}, {nan, nan, nan}, {1.f, 1.f, 1.f}});
  ASSERT_EQ(out.size(), 1u);
  EXPECT_FLOAT_EQ(out[0].x, 1.f);
  EXPECT_EQ(stats.non_finite, 4u);
  EXPECT_EQ(stats.out_of_range, 0u);
}

TEST_P(FilterCases, RangeBoundsAreInclusive) {
  const float below_min = std::nextafter(0.3f, 0.f);
  const float above_max = std::nextafter(10.f, 20.f);
  const auto out = run({{0.3f, 0.f, 0.f},        // r == min_range exactly: kept
                        {0.f, 10.f, 0.f},        // r == max_range exactly: kept
                        {6.f, 0.f, 8.f},         // 6-8-10 triangle, r == 10 exactly: kept
                        {below_min, 0.f, 0.f},   // one ulp inside min: dropped
                        {0.f, 0.f, above_max}}); // one ulp past max: dropped
  EXPECT_EQ(out.size(), 3u);
  EXPECT_EQ(stats.out_of_range, 2u);
}

TEST_P(FilterCases, TwoPointsInOneVoxelGiveTheirMean) {
  const auto out = run({{1.01f, 1.01f, 1.01f}, {1.03f, 1.02f, 1.04f}});
  ASSERT_EQ(out.size(), 1u);
  EXPECT_NEAR(out[0].x, 1.02, 1e-6);
  EXPECT_NEAR(out[0].y, 1.015, 1e-6);
  EXPECT_NEAR(out[0].z, 1.025, 1e-6);
}

TEST_P(FilterCases, VoxelBoundaryFollowsFloor) {
  // voxel_size 0.05: x = -0.0001 -> index -1, x = 0.049 -> index 0, x = 0.05 -> index 1.
  // Three different voxels, emitted in ascending x index even though the input is shuffled.
  const auto out = run({{0.05f, 1.f, 1.f}, {-0.0001f, 1.f, 1.f}, {0.049f, 1.f, 1.f}});
  ASSERT_EQ(out.size(), 3u);
  EXPECT_FLOAT_EQ(out[0].x, -0.0001f);
  EXPECT_FLOAT_EQ(out[1].x, 0.049f);
  EXPECT_FLOAT_EQ(out[2].x, 0.05f);
}

TEST_P(FilterCases, OutputIsInAscendingKeyOrder) {
  // One point at the centre of every voxel of a 5x5x5 block spanning negative and positive
  // indices, shuffled. Expected output: the same centres in lexicographic (ix, iy, iz) order.
  FilterParams p;
  p.voxel_size = 0.5f;
  p.min_range = 0.f;
  std::vector<std::array<int, 3>> idx;
  for (int ix = -2; ix <= 2; ++ix)
    for (int iy = -2; iy <= 2; ++iy)
      for (int iz = -2; iz <= 2; ++iz) idx.push_back({ix, iy, iz});
  auto centre = [&](int i) { return (static_cast<float>(i) + 0.5f) * p.voxel_size; };

  auto shuffled = idx;
  std::mt19937 rng(7);
  std::shuffle(shuffled.begin(), shuffled.end(), rng);
  std::vector<PointXYZ> in;
  for (const auto& i : shuffled) in.push_back({centre(i[0]), centre(i[1]), centre(i[2])});

  const auto out = run(in, p);
  ASSERT_EQ(out.size(), idx.size());
  for (std::size_t k = 0; k < idx.size(); ++k) {
    EXPECT_FLOAT_EQ(out[k].x, centre(idx[k][0])) << "k=" << k;
    EXPECT_FLOAT_EQ(out[k].y, centre(idx[k][1])) << "k=" << k;
    EXPECT_FLOAT_EQ(out[k].z, centre(idx[k][2])) << "k=" << k;
  }
}

TEST_P(FilterCases, KnownGridHasHandComputedCentroids) {
  // 4x4x1 points, voxel_size 0.1. x in {1.02, 1.07 | 1.12, 1.17} -> indices {10 | 11},
  // y in {0.02, 0.07 | 0.12, 0.17} -> indices {0 | 1}, z = 1.03 -> index 10.
  // Four voxels of four points each; means are x in {1.045, 1.145}, y in {0.045, 0.145}.
  FilterParams p;
  p.voxel_size = 0.1f;
  std::vector<PointXYZ> in;
  for (int i = 3; i >= 0; --i)
    for (int j = 0; j < 4; ++j) in.push_back({1.02f + 0.05f * i, 0.02f + 0.05f * j, 1.03f});

  const auto out = run(in, p);
  const double expected[4][3] = {
      {1.045, 0.045, 1.03}, {1.045, 0.145, 1.03}, {1.145, 0.045, 1.03}, {1.145, 0.145, 1.03}};
  ASSERT_EQ(out.size(), 4u);
  for (int k = 0; k < 4; ++k) {
    EXPECT_NEAR(out[k].x, expected[k][0], 1e-5) << "k=" << k;
    EXPECT_NEAR(out[k].y, expected[k][1], 1e-5) << "k=" << k;
    EXPECT_NEAR(out[k].z, expected[k][2], 1e-5) << "k=" << k;
  }
}

TEST_P(FilterCases, IndexOverflowIsDroppedAndCounted) {
  // voxel_size 1e-6: x = 5 m -> index 5e6, outside [-2^20, 2^20). x = 0.5 m -> index ~5e5, kept.
  FilterParams p;
  p.voxel_size = 1e-6f;
  const auto out = run({{5.f, 0.f, 0.f}, {0.5f, 0.f, 0.f}, {0.f, -5.f, 0.f}}, p);
  EXPECT_EQ(out.size(), 1u);
  EXPECT_EQ(stats.index_overflow, 2u);
}

TEST_P(FilterCases, StatsAccountForEveryPoint) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const auto out = run({{nan, 0.f, 0.f}, {0.f, 0.f, 0.f}, {1.f, 1.f, 1.f}, {1.01f, 1.01f, 1.01f}, {2.f, 2.f, 2.f}});
  EXPECT_EQ(out.size(), 2u);
  EXPECT_EQ(stats.non_finite, 1u);
  EXPECT_EQ(stats.out_of_range, 1u);
  EXPECT_EQ(stats.index_overflow, 0u);
}

TEST_P(FilterCases, FilterIsReusableAcrossCalls) {
  auto filter = GetParam().make({});
  std::vector<PointXYZ> out;
  const std::vector<PointXYZ> big{{1.f, 1.f, 1.f}, {2.f, 2.f, 2.f}, {3.f, 3.f, 3.f}};
  const std::vector<PointXYZ> small{{4.f, 4.f, 4.f}};
  filter->apply(big.data(), big.size(), out);
  EXPECT_EQ(out.size(), 3u);
  filter->apply(small.data(), small.size(), out);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_FLOAT_EQ(out[0].x, 4.f);
}

TEST_P(FilterCases, InvalidParamsThrow) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const FilterParams bad[] = {
      {0.f, 0.3f, 10.f},   {-0.05f, 0.3f, 10.f}, {nan, 0.3f, 10.f},  {0.05f, -1.f, 10.f},
      {0.05f, 5.f, 1.f},   {0.05f, nan, 10.f},   {0.05f, 0.3f, nan},
      {0.05f, 0.3f, std::numeric_limits<float>::infinity()}};
  for (const auto& p : bad) {
    EXPECT_THROW(GetParam().make(p), std::invalid_argument)
        << "voxel_size=" << p.voxel_size << " min=" << p.min_range << " max=" << p.max_range;
  }
}

}  // namespace gpu_pointcloud::cases
