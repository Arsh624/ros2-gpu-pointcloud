// bench_h2d: measures host-to-device transfer time as a function of how the source buffer is
// allocated. No filter involved.
//
//   bench_h2d [MEGABYTES]        (default 11, the size of a packed 1280x720 cloud)
//
// For each allocation method it allocates REPEATS separate buffers, touches every page, and
// reports per buffer: the median of 50 synchronous cudaMemcpy calls, the buffer address modulo
// 2 MiB, and how much of the buffer the kernel backs with transparent huge pages.
//
// Prints CSV rows: method,buffer,megabytes,median_ms,gb_per_s,addr_mod_2mib,thp_kb
#include <cuda_runtime.h>
#include <sys/mman.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr std::size_t kHuge = 2u << 20;
constexpr int kRepeats = 6;
constexpr int kCopies = 50;

// Sums AnonHugePages over the mappings that overlap [addr, addr + bytes).
long thp_kb(const void* addr, std::size_t bytes) {
  std::FILE* f = std::fopen("/proc/self/smaps", "r");
  if (!f) return -1;
  const std::uintptr_t lo = reinterpret_cast<std::uintptr_t>(addr), hi = lo + bytes;
  char line[512];
  bool inside = false;
  long total = 0;
  while (std::fgets(line, sizeof(line), f)) {
    std::uintptr_t a = 0, b = 0;
    if (std::sscanf(line, "%" SCNxPTR "-%" SCNxPTR " ", &a, &b) == 2 && std::strchr(line, '-') < std::strchr(line, ' ')) {
      inside = a < hi && b > lo;
    } else if (inside) {
      long kb = 0;
      if (std::sscanf(line, "AnonHugePages: %ld kB", &kb) == 1) total += kb;
    }
  }
  std::fclose(f);
  return total;
}

double median_copy_ms(void* device, const void* host, std::size_t bytes) {
  std::vector<double> ms;
  for (int i = 0; i < kCopies; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice);
    const auto t1 = std::chrono::steady_clock::now();
    ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  std::sort(ms.begin(), ms.end());
  return ms[ms.size() / 2];
}

void report(const char* method, int buffer, void* device, void* host, std::size_t bytes) {
  std::memset(host, 1, bytes);  // touch every page
  const double ms = median_copy_ms(device, host, bytes);
  std::printf("%s,%d,%.1f,%.4f,%.2f,%zu,%ld\n", method, buffer, bytes / 1048576.0, ms, bytes / (ms * 1e6),
              static_cast<std::size_t>(reinterpret_cast<std::uintptr_t>(host) % kHuge), thp_kb(host, bytes));
}

}  // namespace

int main(int argc, char** argv) {
  const std::size_t bytes = static_cast<std::size_t>((argc > 1 ? std::atof(argv[1]) : 11.0) * 1048576.0);
  const std::size_t rounded = (bytes + kHuge - 1) / kHuge * kHuge;
  void* device = nullptr;
  if (cudaMalloc(&device, bytes) != cudaSuccess) {
    std::fprintf(stderr, "error: cudaMalloc failed (no CUDA device?)\n");
    return 1;
  }
  std::vector<char> warm(bytes, 1);
  for (int i = 0; i < 20; ++i) cudaMemcpy(device, warm.data(), bytes, cudaMemcpyHostToDevice);

  std::printf("method,buffer,megabytes,median_ms,gb_per_s,addr_mod_2mib,thp_kb\n");
  std::vector<void*> keep;  // buffers stay allocated so each one gets a new address
  for (int i = 0; i < kRepeats; ++i) {
    void* p = std::malloc(bytes);
    report("malloc", i, device, p, bytes);
    keep.push_back(p);
  }
  for (int i = 0; i < kRepeats; ++i) {
    void* p = std::aligned_alloc(4096, rounded);
    madvise(p, rounded, MADV_NOHUGEPAGE);
    report("no_huge_pages", i, device, p, bytes);
    keep.push_back(p);
  }
  for (int i = 0; i < kRepeats; ++i) {
    void* p = std::aligned_alloc(kHuge, rounded);
    madvise(p, rounded, MADV_HUGEPAGE);
    report("huge_pages", i, device, p, bytes);
    keep.push_back(p);
  }
  for (int i = 0; i < kRepeats; ++i) {
    void* p = nullptr;
    if (cudaMallocHost(&p, bytes) != cudaSuccess) break;
    report("pinned", i, device, p, bytes);
  }
  return 0;
}
