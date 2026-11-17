// Multi-threaded CPU backend (OpenMP). Same algorithm and same output as filter_cpu.cpp:
//   1. classify every point in parallel (static chunks),
//   2. compact the kept (key, index) pairs in input order using per-thread counts,
//   3. sort them with libstdc++'s parallel mode sort (multiway mergesort on OpenMP),
//   4. reduce runs of equal keys in parallel, one contiguous slice of whole runs per thread.
// Pairs are unique, so the sort order is total and the per-voxel summation order is the input
// order, exactly as in the serial backend. That is why the two match bit for bit.
#include "gpu_pointcloud/filter.hpp"

#include <omp.h>
#include <parallel/algorithm>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

#include "gpu_pointcloud/detail/voxel_math.hpp"

namespace gpu_pointcloud {

int default_cpu_threads() { return omp_get_max_threads(); }

namespace {

using Keyed = std::pair<std::uint64_t, std::uint32_t>;

class CpuOmpFilter final : public Filter {
 public:
  CpuOmpFilter(const FilterParams& params, int threads) : params_(params), threads_(threads) {
    validate(params_);
    if (threads_ < 0) throw std::invalid_argument("thread count must be >= 0");
    if (threads_ == 0) threads_ = default_cpu_threads();
  }

  FilterStats apply(const CloudView& view, std::vector<PointXYZ>& out) override {
    validate(view);
    const std::size_t n = view.count;
    const detail::PointReader in(view);
    if (n > std::numeric_limits<std::uint32_t>::max()) {
      throw std::length_error("cloud has more than 2^32 - 1 points");
    }
    FilterStats stats;
    stats.in = n;
    out.clear();
    if (n == 0) return stats;

    const int T = threads_;
    omp_set_num_threads(T);  // also governs __gnu_parallel::sort below
    keys_.resize(n);
    kept_.resize(n);
    std::vector<std::size_t> offset(T + 1, 0);
    std::size_t non_finite = 0, out_of_range = 0, index_overflow = 0;
    const float voxel_size = params_.voxel_size;
    const double min_range = params_.min_range, max_range = params_.max_range;

    // Work is split into T fixed chunks and the loops run over chunk numbers, so the result
    // does not depend on how many threads the runtime actually grants.
#pragma omp parallel for num_threads(T) schedule(static, 1) reduction(+ : non_finite, out_of_range, index_overflow)
    for (int t = 0; t < T; ++t) {
      const std::size_t lo = n * t / T, hi = n * (t + 1) / T;
      std::size_t kept = 0;
      for (std::size_t i = lo; i < hi; ++i) {
        std::uint64_t key = 0;
        const detail::PointClass c = detail::classify(in(i), voxel_size, min_range, max_range, key);
        keys_[i] = key;
        kept_[i] = c == detail::kKept;
        kept += c == detail::kKept;
        non_finite += c == detail::kNonFinite;
        out_of_range += c == detail::kOutOfRange;
        index_overflow += c == detail::kIndexOverflow;
      }
      offset[t + 1] = kept;
    }
    for (int t = 0; t < T; ++t) offset[t + 1] += offset[t];
    keyed_.resize(offset[T]);
#pragma omp parallel for num_threads(T) schedule(static, 1)
    for (int t = 0; t < T; ++t) {
      const std::size_t lo = n * t / T, hi = n * (t + 1) / T;
      std::size_t j = offset[t];
      for (std::size_t i = lo; i < hi; ++i) {
        if (kept_[i]) keyed_[j++] = {keys_[i], static_cast<std::uint32_t>(i)};
      }
    }
    stats.non_finite = non_finite;
    stats.out_of_range = out_of_range;
    stats.index_overflow = index_overflow;

    const std::size_t m = keyed_.size();
    __gnu_parallel::sort(keyed_.begin(), keyed_.end());

    // Slice boundaries moved forward to the next run start, so no run is split between threads.
    std::vector<std::size_t> begin(T + 1), runs(T + 1, 0);
    for (int t = 0; t <= T; ++t) {
      std::size_t b = m * t / T;
      while (b > 0 && b < m && keyed_[b].first == keyed_[b - 1].first) ++b;
      begin[t] = b;
    }
#pragma omp parallel for num_threads(T) schedule(static, 1)
    for (int t = 0; t < T; ++t) {
      std::size_t count = 0;
      for (std::size_t i = begin[t]; i < begin[t + 1]; ++i) {
        count += i == begin[t] || keyed_[i].first != keyed_[i - 1].first;
      }
      runs[t + 1] = count;
    }
    for (int t = 0; t < T; ++t) runs[t + 1] += runs[t];
    out.resize(runs[T]);

#pragma omp parallel for num_threads(T) schedule(static, 1)
    for (int t = 0; t < T; ++t) {
      std::size_t o = runs[t];
      for (std::size_t i = begin[t]; i < begin[t + 1];) {
        const std::uint64_t key = keyed_[i].first;
        double sx = 0.0, sy = 0.0, sz = 0.0;
        std::size_t count = 0;
        for (; i < begin[t + 1] && keyed_[i].first == key; ++i, ++count) {
          const PointXYZ p = in(keyed_[i].second);
          sx += p.x;
          sy += p.y;
          sz += p.z;
        }
        const double inv = 1.0 / static_cast<double>(count);
        out[o++] = {static_cast<float>(sx * inv), static_cast<float>(sy * inv), static_cast<float>(sz * inv)};
      }
    }
    stats.out = out.size();
    return stats;
  }

 private:
  FilterParams params_;
  int threads_;
  // Reused across frames.
  std::vector<std::uint64_t> keys_;
  std::vector<std::uint8_t> kept_;
  std::vector<Keyed> keyed_;
};

}  // namespace

std::unique_ptr<Filter> make_cpu_omp_filter(const FilterParams& params, int threads) {
  return std::make_unique<CpuOmpFilter>(params, threads);
}

}  // namespace gpu_pointcloud
