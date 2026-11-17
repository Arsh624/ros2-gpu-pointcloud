// CUDA backend. Pipeline per frame, all on one stream with preallocated buffers:
//
//   H2D copy -> classify kernel (key + keep flag + reject counters)
//            -> cub::DeviceSelect::Flagged   (indices of kept points)
//            -> gather kernel                (keys of kept points)
//            -> cub::DeviceRadixSort         (sort (key, index) by the used key bits only)
//            -> cub::DeviceReduce::ReduceByKey (sum x, y, z, count per voxel, in double)
//            -> divide kernel                (centroids as float32)
//            -> D2H copy
//
// See docs/DESIGN.md for why each stage looks the way it does.
#include "gpu_pointcloud/filter.hpp"

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "gpu_pointcloud/detail/voxel_math.hpp"

namespace gpu_pointcloud {

bool cuda_device_present() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

namespace {

#define GPC_CUDA_CHECK(call)                                                                      \
  do {                                                                                            \
    const cudaError_t gpc_err = (call);                                                           \
    if (gpc_err != cudaSuccess) {                                                                 \
      throw std::runtime_error(std::string("CUDA error in " #call ": ") + cudaGetErrorString(gpc_err)); \
    }                                                                                             \
  } while (0)

struct NvtxRange {
  explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
  ~NvtxRange() { nvtxRangePop(); }
};

constexpr int kBlock = 256;

// How the canonical 21-bit-per-axis key is narrowed for sorting. Every kept point satisfies
// |coord| <= max_range, so its per-axis index lies in [-limit, limit]. Subtracting
// (2^20 - limit) from each field and repacking with `bits` per axis keeps the ordering of the
// canonical key but leaves only 3 * bits significant bits for the radix sort.
struct KeyLayout {
  std::uint64_t rebase;  // subtracted from each canonical field
  int bits;              // bits per axis after rebasing
};

KeyLayout make_key_layout(const FilterParams& p) {
  const double limit = std::ceil(static_cast<double>(p.max_range / p.voxel_size)) + 1.0;
  if (limit < static_cast<double>(detail::kIndexLimit)) {
    int bits = 1;
    while ((1ull << bits) < 2 * static_cast<std::uint64_t>(limit) + 1) ++bits;
    if (bits < detail::kAxisBits) {
      return {static_cast<std::uint64_t>(detail::kIndexOffset) - static_cast<std::uint64_t>(limit), bits};
    }
  }
  return {0, detail::kAxisBits};  // canonical key as is
}

// Reject counters are indexed by PointClass - 1.
struct Meta {
  unsigned int rejected[3];
  int num_kept;
  int num_voxels;
};

__global__ void classify_kernel(const PointXYZ* points, int n, float voxel_size, double min_range,
                                double max_range, KeyLayout layout, std::uint64_t* keys,
                                std::uint8_t* keep, unsigned int* rejected) {
  __shared__ unsigned int block_rejected[3];
  if (threadIdx.x < 3) block_rejected[threadIdx.x] = 0;
  __syncthreads();

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    std::uint64_t key = 0;
    const detail::PointClass c = detail::classify(points[i], voxel_size, min_range, max_range, key);
    if (c == detail::kKept) {
      const std::uint64_t mask = (1ull << detail::kAxisBits) - 1;
      const std::uint64_t ix = (key >> (2 * detail::kAxisBits)) - layout.rebase;
      const std::uint64_t iy = ((key >> detail::kAxisBits) & mask) - layout.rebase;
      const std::uint64_t iz = (key & mask) - layout.rebase;
      keys[i] = (ix << (2 * layout.bits)) | (iy << layout.bits) | iz;
      keep[i] = 1;
    } else {
      keep[i] = 0;
      atomicAdd(&block_rejected[c - 1], 1u);
    }
  }
  __syncthreads();
  // One global atomic per block and class instead of one per rejected point.
  if (threadIdx.x < 3 && block_rejected[threadIdx.x] != 0) {
    atomicAdd(&rejected[threadIdx.x], block_rejected[threadIdx.x]);
  }
}

__global__ void gather_keys_kernel(const std::uint64_t* keys, const std::uint32_t* index, int m,
                                   std::uint64_t* out) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < m) out[j] = keys[index[j]];
}

struct Sum {
  double x, y, z, n;
};

struct LoadPoint {
  const PointXYZ* points;
  __host__ __device__ Sum operator()(std::uint32_t i) const {
    const PointXYZ p = points[i];
    return {static_cast<double>(p.x), static_cast<double>(p.y), static_cast<double>(p.z), 1.0};
  }
};

struct AddSum {
  __host__ __device__ Sum operator()(const Sum& a, const Sum& b) const {
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.n + b.n};
  }
};

__global__ void centroid_kernel(const Sum* sums, int m, PointXYZ* out) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < m) {
    const Sum s = sums[j];
    const double inv = 1.0 / s.n;
    out[j] = {static_cast<float>(s.x * inv), static_cast<float>(s.y * inv), static_cast<float>(s.z * inv)};
  }
}

int blocks_for(int n) { return (n + kBlock - 1) / kBlock; }

class CudaFilter final : public Filter {
 public:
  CudaFilter(const FilterParams& params, std::size_t reserve_points)
      : params_(params), layout_(make_key_layout(params)) {
    GPC_CUDA_CHECK(cudaStreamCreate(&stream_));
    for (cudaEvent_t& ev : ev_) GPC_CUDA_CHECK(cudaEventCreate(&ev));
    GPC_CUDA_CHECK(cudaMalloc(&d_meta_, sizeof(Meta)));
    GPC_CUDA_CHECK(cudaMallocHost(&h_meta_, sizeof(Meta)));
    reserve(std::max<std::size_t>(reserve_points, 1));
  }

  ~CudaFilter() override {
    release_buffers();
    cudaFree(d_meta_);
    cudaFreeHost(h_meta_);
    cudaFree(d_temp_);
    for (cudaEvent_t ev : ev_) if (ev) cudaEventDestroy(ev);
    if (stream_) cudaStreamDestroy(stream_);
  }

  CudaFilter(const CudaFilter&) = delete;
  CudaFilter& operator=(const CudaFilter&) = delete;

  FilterStats apply(const PointXYZ* in, std::size_t n_points, std::vector<PointXYZ>& out) override {
    if (n_points > static_cast<std::size_t>(INT_MAX)) throw std::length_error("cloud has more than 2^31 - 1 points");
    FilterStats stats;
    stats.in = n_points;
    out.clear();
    if (n_points == 0) return stats;
    if (n_points > capacity_) reserve(n_points);
    const int n = static_cast<int>(n_points);
    mark(kBegin);

    {
      NvtxRange r("h2d");
      std::memcpy(h_in_, in, n_points * sizeof(PointXYZ));
      GPC_CUDA_CHECK(cudaMemcpyAsync(d_points_, h_in_, n_points * sizeof(PointXYZ), cudaMemcpyHostToDevice, stream_));
    }
    mark(kAfterH2D);

    int num_kept = 0;
    {
      NvtxRange r("classify+select");
      GPC_CUDA_CHECK(cudaMemsetAsync(d_meta_, 0, sizeof(Meta), stream_));
      classify_kernel<<<blocks_for(n), kBlock, 0, stream_>>>(d_points_, n, params_.voxel_size, params_.min_range,
                                                             params_.max_range, layout_, d_keys_a_, d_keep_,
                                                             d_meta_->rejected);
      GPC_CUDA_CHECK(cudaGetLastError());

      mark(kAfterClassify);
      cub::CountingInputIterator<std::uint32_t> all_indices(0);
      std::size_t bytes = 0;
      GPC_CUDA_CHECK(cub::DeviceSelect::Flagged(nullptr, bytes, all_indices, d_keep_, d_index_a_,
                                                &d_meta_->num_kept, n, stream_));
      ensure_temp(bytes);
      GPC_CUDA_CHECK(cub::DeviceSelect::Flagged(d_temp_, bytes, all_indices, d_keep_, d_index_a_,
                                                &d_meta_->num_kept, n, stream_));
      mark(kAfterSelect);
      // The host needs the kept count to size the sort, so this is a required sync point.
      fetch_meta();
      num_kept = h_meta_->num_kept;
    }
    stats.non_finite = h_meta_->rejected[detail::kNonFinite - 1];
    stats.out_of_range = h_meta_->rejected[detail::kOutOfRange - 1];
    stats.index_overflow = h_meta_->rejected[detail::kIndexOverflow - 1];

    int num_voxels = 0;
    if (num_kept > 0) {
      {
        NvtxRange r("gather_keys");
        gather_keys_kernel<<<blocks_for(num_kept), kBlock, 0, stream_>>>(d_keys_a_, d_index_a_, num_kept, d_keys_b_);
        GPC_CUDA_CHECK(cudaGetLastError());
      }
      mark(kAfterGather);
      // d_keys_a_ is free again after the gather, so it serves as the sort's alternate buffer.
      cub::DoubleBuffer<std::uint64_t> keys(d_keys_b_, d_keys_a_);
      cub::DoubleBuffer<std::uint32_t> index(d_index_a_, d_index_b_);
      {
        NvtxRange r("sort");
        std::size_t bytes = 0;
        GPC_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, bytes, keys, index, num_kept, 0, 3 * layout_.bits, stream_));
        ensure_temp(bytes);
        GPC_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(d_temp_, bytes, keys, index, num_kept, 0, 3 * layout_.bits, stream_));
      }
      mark(kAfterSort);
      {
        NvtxRange r("reduce_by_key");
        cub::TransformInputIterator<Sum, LoadPoint, const std::uint32_t*> values(index.Current(), LoadPoint{d_points_});
        cub::DiscardOutputIterator<> unique_keys;
        std::size_t bytes = 0;
        GPC_CUDA_CHECK(cub::DeviceReduce::ReduceByKey(nullptr, bytes, keys.Current(), unique_keys, values, d_sums_,
                                                      &d_meta_->num_voxels, AddSum{}, num_kept, stream_));
        ensure_temp(bytes);
        GPC_CUDA_CHECK(cub::DeviceReduce::ReduceByKey(d_temp_, bytes, keys.Current(), unique_keys, values, d_sums_,
                                                      &d_meta_->num_voxels, AddSum{}, num_kept, stream_));
        mark(kAfterReduce);
        // Second required sync point: the host needs the voxel count to size the output copy.
        fetch_meta();
        num_voxels = h_meta_->num_voxels;
      }
      {
        NvtxRange r("centroid");
        centroid_kernel<<<blocks_for(num_voxels), kBlock, 0, stream_>>>(d_sums_, num_voxels, d_out_);
        GPC_CUDA_CHECK(cudaGetLastError());
      }
    } else {
      // Nothing survived the filter: keep the skipped stage boundaries valid (zero-length stages).
      mark(kAfterGather);
      mark(kAfterSort);
      mark(kAfterReduce);
    }
    mark(kAfterCentroid);

    {
      NvtxRange r("d2h");
      if (num_voxels > 0) {
        GPC_CUDA_CHECK(cudaMemcpyAsync(h_out_, d_out_, num_voxels * sizeof(PointXYZ), cudaMemcpyDeviceToHost, stream_));
      }
      mark(kAfterD2H);
      GPC_CUDA_CHECK(cudaStreamSynchronize(stream_));
      out.resize(num_voxels);
      if (num_voxels > 0) std::memcpy(out.data(), h_out_, num_voxels * sizeof(PointXYZ));
    }

    for (int i = 0; i + 1 < kNumMarks; ++i) stage_ms_[i] = elapsed_ms(i, i + 1);
    stats.device_ms = elapsed_ms(kAfterH2D, kAfterCentroid);
    stats.out = out.size();
    return stats;
  }

  std::vector<StageTime> last_stage_times() const override {
    // Each stage is the interval between two events on the stream, so it is GPU-clock time.
    // The two stages that follow a host sync also contain the host round trip.
    static const char* const names[kNumMarks - 1] = {
        "h2d_copy", "classify", "select", "host_sync+gather_keys", "radix_sort",
        "reduce_by_key", "host_sync+centroid", "d2h_copy"};
    std::vector<StageTime> stages;
    for (int i = 0; i + 1 < kNumMarks; ++i) stages.push_back({names[i], stage_ms_[i]});
    return stages;
  }

 private:
  enum Mark {
    kBegin, kAfterH2D, kAfterClassify, kAfterSelect, kAfterGather, kAfterSort, kAfterReduce,
    kAfterCentroid, kAfterD2H, kNumMarks
  };

  void mark(Mark m) { GPC_CUDA_CHECK(cudaEventRecord(ev_[m], stream_)); }

  double elapsed_ms(int from, int to) const {
    float ms = 0.0f;
    GPC_CUDA_CHECK(cudaEventElapsedTime(&ms, ev_[from], ev_[to]));
    return ms;
  }

  void fetch_meta() {
    GPC_CUDA_CHECK(cudaMemcpyAsync(h_meta_, d_meta_, sizeof(Meta), cudaMemcpyDeviceToHost, stream_));
    GPC_CUDA_CHECK(cudaStreamSynchronize(stream_));
  }

  void ensure_temp(std::size_t bytes) {
    if (bytes <= temp_bytes_) return;
    GPC_CUDA_CHECK(cudaStreamSynchronize(stream_));
    cudaFree(d_temp_);
    d_temp_ = nullptr;
    temp_bytes_ = 0;
    GPC_CUDA_CHECK(cudaMalloc(&d_temp_, bytes));
    temp_bytes_ = bytes;
  }

  void release_buffers() {
    cudaFree(d_points_);
    cudaFree(d_keys_a_);
    cudaFree(d_keys_b_);
    cudaFree(d_keep_);
    cudaFree(d_index_a_);
    cudaFree(d_index_b_);
    cudaFree(d_sums_);
    cudaFree(d_out_);
    cudaFreeHost(h_in_);
    cudaFreeHost(h_out_);
    d_points_ = nullptr;
    d_keys_a_ = d_keys_b_ = nullptr;
    d_keep_ = nullptr;
    d_index_a_ = d_index_b_ = nullptr;
    d_sums_ = nullptr;
    d_out_ = nullptr;
    h_in_ = h_out_ = nullptr;
    capacity_ = 0;
  }

  // Allocates every per-point buffer for `n` points and sizes the CUB temp storage for the
  // worst case, so apply() does not allocate for clouds up to this size.
  void reserve(std::size_t n) {
    GPC_CUDA_CHECK(cudaStreamSynchronize(stream_));
    release_buffers();
    GPC_CUDA_CHECK(cudaMalloc(&d_points_, n * sizeof(PointXYZ)));
    GPC_CUDA_CHECK(cudaMalloc(&d_keys_a_, n * sizeof(std::uint64_t)));
    GPC_CUDA_CHECK(cudaMalloc(&d_keys_b_, n * sizeof(std::uint64_t)));
    GPC_CUDA_CHECK(cudaMalloc(&d_keep_, n * sizeof(std::uint8_t)));
    GPC_CUDA_CHECK(cudaMalloc(&d_index_a_, n * sizeof(std::uint32_t)));
    GPC_CUDA_CHECK(cudaMalloc(&d_index_b_, n * sizeof(std::uint32_t)));
    GPC_CUDA_CHECK(cudaMalloc(&d_sums_, n * sizeof(Sum)));
    GPC_CUDA_CHECK(cudaMalloc(&d_out_, n * sizeof(PointXYZ)));
    GPC_CUDA_CHECK(cudaMallocHost(&h_in_, n * sizeof(PointXYZ)));
    GPC_CUDA_CHECK(cudaMallocHost(&h_out_, n * sizeof(PointXYZ)));
    capacity_ = n;

    // Size queries only (null temp storage): nothing runs on the device here.
    const int count = static_cast<int>(std::min<std::size_t>(n, INT_MAX));
    std::size_t select_bytes = 0, sort_bytes = 0, reduce_bytes = 0;
    cub::CountingInputIterator<std::uint32_t> all_indices(0);
    GPC_CUDA_CHECK(cub::DeviceSelect::Flagged(nullptr, select_bytes, all_indices, d_keep_, d_index_a_,
                                              &d_meta_->num_kept, count, stream_));
    cub::DoubleBuffer<std::uint64_t> keys(d_keys_b_, d_keys_a_);
    cub::DoubleBuffer<std::uint32_t> index(d_index_a_, d_index_b_);
    GPC_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, keys, index, count, 0, 3 * layout_.bits, stream_));
    cub::TransformInputIterator<Sum, LoadPoint, const std::uint32_t*> values(d_index_a_, LoadPoint{d_points_});
    cub::DiscardOutputIterator<> unique_keys;
    GPC_CUDA_CHECK(cub::DeviceReduce::ReduceByKey(nullptr, reduce_bytes, d_keys_b_, unique_keys, values, d_sums_,
                                                  &d_meta_->num_voxels, AddSum{}, count, stream_));
    ensure_temp(std::max({select_bytes, sort_bytes, reduce_bytes}));
  }

  FilterParams params_;
  KeyLayout layout_;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t ev_[kNumMarks] = {};
  double stage_ms_[kNumMarks - 1] = {};

  std::size_t capacity_ = 0;  // points the buffers below can hold
  PointXYZ* d_points_ = nullptr;
  std::uint64_t* d_keys_a_ = nullptr;  // keys of all points, then the sort's alternate buffer
  std::uint64_t* d_keys_b_ = nullptr;  // keys of kept points
  std::uint8_t* d_keep_ = nullptr;
  std::uint32_t* d_index_a_ = nullptr;  // indices of kept points
  std::uint32_t* d_index_b_ = nullptr;  // sort's alternate buffer
  Sum* d_sums_ = nullptr;
  PointXYZ* d_out_ = nullptr;
  PointXYZ* h_in_ = nullptr;   // pinned
  PointXYZ* h_out_ = nullptr;  // pinned

  Meta* d_meta_ = nullptr;
  Meta* h_meta_ = nullptr;  // pinned
  void* d_temp_ = nullptr;  // CUB temp storage, shared by the three CUB stages
  std::size_t temp_bytes_ = 0;
};

}  // namespace

std::unique_ptr<Filter> make_cuda_filter(const FilterParams& params, std::size_t reserve_points) {
  validate(params);
  if (!cuda_device_present()) {
    throw std::runtime_error("backend 'cuda' requested but no CUDA device is visible");
  }
  return std::make_unique<CudaFilter>(params, reserve_points);
}

}  // namespace gpu_pointcloud
