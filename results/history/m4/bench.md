iterations: 200 timed after 20 warmup, voxel_size=0.05 m, range 0.3 to 10.0 m

| resolution | points | backend | timing | median ms | p99 ms | input points/s | output points |
|---|---:|---|---|---:|---:|---:|---:|
| 320x240 | 76800 | cpu | end_to_end | 1.825 | 2.094 | 4.21e+07 | 20785 |
| 320x240 | 76800 | cuda | end_to_end | 0.461 | 0.624 | 1.67e+08 | 20785 |
| 320x240 | 76800 | cuda | gpu_only | 0.232 | 0.365 | 3.31e+08 | 20785 |
| 640x480 | 307200 | cpu | end_to_end | 7.914 | 9.462 | 3.88e+07 | 34654 |
| 640x480 | 307200 | cuda | end_to_end | 1.049 | 1.313 | 2.93e+08 | 34654 |
| 640x480 | 307200 | cuda | gpu_only | 0.370 | 0.592 | 8.3e+08 | 34654 |
| 1280x720 | 921600 | cpu | end_to_end | 25.683 | 31.828 | 3.59e+07 | 39514 |
| 1280x720 | 921600 | cuda | end_to_end | 2.439 | 12.256 | 3.78e+08 | 39514 |
| 1280x720 | 921600 | cuda | gpu_only | 0.586 | 3.738 | 1.57e+09 | 39514 |

Speedup = CPU median / CUDA median.

| resolution | vs CUDA end_to_end | vs CUDA gpu_only |
|---|---:|---:|
| 320x240 | 3.96x | 7.86x |
| 640x480 | 7.55x | 21.37x |
| 1280x720 | 10.53x | 43.80x |

CUDA stages, median ms per stage (cudaEvent intervals on the stream).

| resolution | stage | median ms |
|---|---|---:|
| 320x240 | h2d_copy | 0.140 |
| 320x240 | classify | 0.044 |
| 320x240 | select | 0.010 |
| 320x240 | host_sync+gather_keys | 0.034 |
| 320x240 | radix_sort | 0.085 |
| 320x240 | reduce_by_key | 0.029 |
| 320x240 | host_sync+centroid | 0.031 |
| 320x240 | d2h_copy | 0.044 |
| 640x480 | h2d_copy | 0.555 |
| 640x480 | classify | 0.121 |
| 640x480 | select | 0.012 |
| 640x480 | host_sync+gather_keys | 0.042 |
| 640x480 | radix_sort | 0.088 |
| 640x480 | reduce_by_key | 0.069 |
| 640x480 | host_sync+centroid | 0.031 |
| 640x480 | d2h_copy | 0.061 |
| 1280x720 | h2d_copy | 1.657 |
| 1280x720 | classify | 0.196 |
| 1280x720 | select | 0.013 |
| 1280x720 | host_sync+gather_keys | 0.054 |
| 1280x720 | radix_sort | 0.134 |
| 1280x720 | reduce_by_key | 0.123 |
| 1280x720 | host_sync+centroid | 0.037 |
| 1280x720 | d2h_copy | 0.064 |
