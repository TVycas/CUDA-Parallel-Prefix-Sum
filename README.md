 # Parallel Prefix Sum (Scan) with CUDA
 
 This was one of the assignments for my Distributed & Parallel Computing module at the University of Birmingham.


 For this assignment, we wrote a CUDA program that implements a work efficient exclusive scan as described in [GPU Gems 3, Chapter 39](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda) and demonstrated it by applying it to a large vector of integers.

 ---

### Achievements
 - Block scanning
 - Full scan for large vectors (support for second and third level scans)
 - Bank conflict avoidance optimization (BCAO)

 ---
 
## Tests

The block size for the tests was 128 and the vector size was 10000000.

 - Block scan without BCAO = 1.10294 msecs
 - Block scan with BCAO = 0.47206 msecs
 - Full scan without BCAO = 1.39594 msecs
 - Full scan with BCAO = 0.76058 msecs

### Machine:
 - CPU - Intel® Core™ i7-8700 CPU @ 3.20GHz × 12
 - GPU - GeForce RTX 2060
