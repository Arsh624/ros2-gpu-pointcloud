iterations: 200 timed after 20 warmup (and at least 500 ms), voxel_size=0.05 m, range 0.3 to 10.0 m, cpu_omp threads: 20, point step: 12 bytes, cuda input: auto

| resolution | points | backend | timing | median ms | p99 ms | input points/s | output points |
|---|---:|---|---|---:|---:|---:|---:|
| 320x240 | 76800 | cpu | end_to_end | 1.894 | 2.174 | 4.05e+07 | 20785 |
| 320x240 | 76800 | cpu_omp | end_to_end | 0.246 | 0.490 | 3.13e+08 | 20785 |
| 320x240 | 76800 | pcl | end_to_end | 1.603 | 1.972 | 4.79e+07 | 20785 |
| 320x240 | 76800 | cuda | end_to_end | 0.415 | 0.676 | 1.85e+08 | 20785 |
| 320x240 | 76800 | cuda | gpu_only | 0.218 | 0.376 | 3.52e+08 | 20785 |
| 640x480 | 307200 | cpu | end_to_end | 8.732 | 12.042 | 3.52e+07 | 34654 |
| 640x480 | 307200 | cpu_omp | end_to_end | 0.914 | 1.583 | 3.36e+08 | 34654 |
| 640x480 | 307200 | pcl | end_to_end | 5.011 | 6.015 | 6.13e+07 | 34654 |
| 640x480 | 307200 | cuda | end_to_end | 0.705 | 0.815 | 4.36e+08 | 34654 |
| 640x480 | 307200 | cuda | gpu_only | 0.256 | 0.339 | 1.2e+09 | 34654 |
| 1280x720 | 921600 | cpu | end_to_end | 26.156 | 29.344 | 3.52e+07 | 39514 |
| 1280x720 | 921600 | cpu_omp | end_to_end | 2.889 | 12.901 | 3.19e+08 | 39514 |
| 1280x720 | 921600 | pcl | end_to_end | 14.242 | 17.667 | 6.47e+07 | 39514 |
| 1280x720 | 921600 | cuda | end_to_end | 1.641 | 1.917 | 5.61e+08 | 39514 |
| 1280x720 | 921600 | cuda | gpu_only | 0.435 | 0.547 | 2.12e+09 | 39514 |

Speedup = CPU median / CUDA median. "Best CPU" is the fastest CPU baseline measured in this run at that resolution.

| resolution | best CPU | best CPU ms | CUDA end_to_end vs best CPU | CUDA gpu_only vs best CPU | CUDA end_to_end vs single-threaded cpu | CUDA gpu_only vs single-threaded cpu |
|---|---|---:|---:|---:|---:|---:|
| 320x240 | cpu_omp | 0.246 | 0.59x | 1.13x | 4.56x | 8.68x |
| 640x480 | cpu_omp | 0.914 | 1.30x | 3.57x | 12.38x | 34.08x |
| 1280x720 | cpu_omp | 2.889 | 1.76x | 6.64x | 15.93x | 60.17x |

CUDA stages, median ms per stage (cudaEvent intervals on the stream).

| resolution | stage | median ms |
|---|---|---:|
| 320x240 | h2d_copy | 0.110 |
| 320x240 | classify | 0.037 |
| 320x240 | select | 0.007 |
| 320x240 | host_sync+gather_keys | 0.033 |
| 320x240 | radix_sort | 0.079 |
| 320x240 | reduce_by_key | 0.022 |
| 320x240 | host_sync+centroid | 0.028 |
| 320x240 | d2h_copy | 0.045 |
| 640x480 | h2d_copy | 0.342 |
| 640x480 | classify | 0.063 |
| 640x480 | select | 0.009 |
| 640x480 | host_sync+gather_keys | 0.028 |
| 640x480 | radix_sort | 0.080 |
| 640x480 | reduce_by_key | 0.047 |
| 640x480 | host_sync+centroid | 0.026 |
| 640x480 | d2h_copy | 0.058 |
| 1280x720 | h2d_copy | 1.053 |
| 1280x720 | classify | 0.140 |
| 1280x720 | select | 0.012 |
| 1280x720 | host_sync+gather_keys | 0.034 |
| 1280x720 | radix_sort | 0.098 |
| 1280x720 | reduce_by_key | 0.114 |
| 1280x720 | host_sync+centroid | 0.032 |
| 1280x720 | d2h_copy | 0.066 |
