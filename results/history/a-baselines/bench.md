iterations: 200 timed after 20 warmup, voxel_size=0.05 m, range 0.3 to 10.0 m, cpu_omp threads: 20

| resolution | points | backend | timing | median ms | p99 ms | input points/s | output points |
|---|---:|---|---|---:|---:|---:|---:|
| 320x240 | 76800 | cpu | end_to_end | 1.918 | 2.295 | 4e+07 | 20785 |
| 320x240 | 76800 | cpu_omp | end_to_end | 0.241 | 0.527 | 3.18e+08 | 20785 |
| 320x240 | 76800 | pcl | end_to_end | 1.574 | 1.810 | 4.88e+07 | 20785 |
| 320x240 | 76800 | cuda | end_to_end | 0.442 | 0.548 | 1.74e+08 | 20785 |
| 320x240 | 76800 | cuda | gpu_only | 0.233 | 0.329 | 3.3e+08 | 20785 |
| 640x480 | 307200 | cpu | end_to_end | 7.995 | 8.882 | 3.84e+07 | 34654 |
| 640x480 | 307200 | cpu_omp | end_to_end | 0.856 | 1.197 | 3.59e+08 | 34654 |
| 640x480 | 307200 | pcl | end_to_end | 4.896 | 5.557 | 6.27e+07 | 34654 |
| 640x480 | 307200 | cuda | end_to_end | 0.998 | 1.161 | 3.08e+08 | 34654 |
| 640x480 | 307200 | cuda | gpu_only | 0.358 | 0.504 | 8.59e+08 | 34654 |
| 1280x720 | 921600 | cpu | end_to_end | 25.788 | 27.121 | 3.57e+07 | 39514 |
| 1280x720 | 921600 | cpu_omp | end_to_end | 2.630 | 3.326 | 3.5e+08 | 39514 |
| 1280x720 | 921600 | pcl | end_to_end | 13.885 | 16.188 | 6.64e+07 | 39514 |
| 1280x720 | 921600 | cuda | end_to_end | 2.356 | 11.405 | 3.91e+08 | 39514 |
| 1280x720 | 921600 | cuda | gpu_only | 0.567 | 3.058 | 1.63e+09 | 39514 |

Speedup = CPU median / CUDA median. "Best CPU" is the fastest CPU baseline measured in this run at that resolution.

| resolution | best CPU | best CPU ms | CUDA end_to_end vs best CPU | CUDA gpu_only vs best CPU | CUDA end_to_end vs single-threaded cpu | CUDA gpu_only vs single-threaded cpu |
|---|---|---:|---:|---:|---:|---:|
| 320x240 | cpu_omp | 0.241 | 0.55x | 1.04x | 4.34x | 8.23x |
| 640x480 | cpu_omp | 0.856 | 0.86x | 2.39x | 8.01x | 22.36x |
| 1280x720 | cpu_omp | 2.630 | 1.12x | 4.64x | 10.95x | 45.48x |

CUDA stages, median ms per stage (cudaEvent intervals on the stream).

| resolution | stage | median ms |
|---|---|---:|
| 320x240 | h2d_copy | 0.127 |
| 320x240 | classify | 0.043 |
| 320x240 | select | 0.010 |
| 320x240 | host_sync+gather_keys | 0.024 |
| 320x240 | radix_sort | 0.088 |
| 320x240 | reduce_by_key | 0.029 |
| 320x240 | host_sync+centroid | 0.028 |
| 320x240 | d2h_copy | 0.040 |
| 640x480 | h2d_copy | 0.517 |
| 640x480 | classify | 0.095 |
| 640x480 | select | 0.012 |
| 640x480 | host_sync+gather_keys | 0.042 |
| 640x480 | radix_sort | 0.092 |
| 640x480 | reduce_by_key | 0.069 |
| 640x480 | host_sync+centroid | 0.031 |
| 640x480 | d2h_copy | 0.054 |
| 1280x720 | h2d_copy | 1.618 |
| 1280x720 | classify | 0.198 |
| 1280x720 | select | 0.014 |
| 1280x720 | host_sync+gather_keys | 0.053 |
| 1280x720 | radix_sort | 0.126 |
| 1280x720 | reduce_by_key | 0.121 |
| 1280x720 | host_sync+centroid | 0.035 |
| 1280x720 | d2h_copy | 0.060 |
