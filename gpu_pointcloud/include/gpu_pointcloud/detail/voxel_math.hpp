// Per-point classification shared by the CPU backend and the CUDA kernel, so both run the same
// arithmetic. Compiled with floating-point contraction off on both sides (see docs/DESIGN.md).
#pragma once

#include <cmath>
#include <cstdint>

#include "gpu_pointcloud/filter.hpp"

#if defined(__CUDACC__)
#define GPC_HOST_DEVICE __host__ __device__
#else
#define GPC_HOST_DEVICE
#endif

namespace gpu_pointcloud::detail {

constexpr int kAxisBits = 21;
constexpr float kIndexLimit = 1048576.0f;  // 2^20: indices live in [-2^20, 2^20)
constexpr std::int64_t kIndexOffset = 1 << 20;

enum PointClass : std::uint8_t { kKept = 0, kNonFinite = 1, kOutOfRange = 2, kIndexOverflow = 3 };

GPC_HOST_DEVICE inline bool is_finite(float v) { return (v - v) == 0.0f; }  // false for NaN and Inf

// Packs floor(coord / voxel_size) per axis. Returns false when an index does not fit in 21 bits.
// A quotient that overflows to infinity is rejected by the same comparison.
GPC_HOST_DEVICE inline bool pack_key(const PointXYZ& p, float voxel_size, std::uint64_t& key) {
  const float fx = std::floor(p.x / voxel_size);
  const float fy = std::floor(p.y / voxel_size);
  const float fz = std::floor(p.z / voxel_size);
  if (!(fx >= -kIndexLimit && fx < kIndexLimit && fy >= -kIndexLimit && fy < kIndexLimit &&
        fz >= -kIndexLimit && fz < kIndexLimit)) {
    return false;
  }
  const std::uint64_t ix = static_cast<std::uint64_t>(static_cast<std::int64_t>(fx) + kIndexOffset);
  const std::uint64_t iy = static_cast<std::uint64_t>(static_cast<std::int64_t>(fy) + kIndexOffset);
  const std::uint64_t iz = static_cast<std::uint64_t>(static_cast<std::int64_t>(fz) + kIndexOffset);
  key = (ix << (2 * kAxisBits)) | (iy << kAxisBits) | iz;
  return true;
}

// Steps 1-4 of the filter for one point. `key` is written only when the result is kKept.
GPC_HOST_DEVICE inline PointClass classify(const PointXYZ& p, float voxel_size, double min_range,
                                           double max_range, std::uint64_t& key) {
  if (!(is_finite(p.x) && is_finite(p.y) && is_finite(p.z))) return kNonFinite;
  const double x = p.x, y = p.y, z = p.z;
  const double r = std::sqrt(x * x + y * y + z * z);
  if (!(r >= min_range && r <= max_range)) return kOutOfRange;
  return pack_key(p, voxel_size, key) ? kKept : kIndexOverflow;
}

}  // namespace gpu_pointcloud::detail
