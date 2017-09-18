## Exercise 4

For implementation, see file "simple.cu"

I typically see such results:
    
    CPU Took 132 microseconds (0.13ms)
    GPU Took 46 microseconds (0.05ms)

The GPU is a factor of ~3 faster than the CPU. Even though the CPU is running sequentially, the faster clock speed and cache control makes it relatively fast at computing the result. The GPU however while faster, does not have a speedup by a factor of the number of cores. The clock speed of each core is of course much lower than the CPU's cores but the explanation is probably that the GPU spends too much time on retrieving/writing to global memory. This means that the math/memory ratio is too low.