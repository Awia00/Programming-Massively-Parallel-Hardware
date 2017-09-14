## Exercise 2

For implementation see file "lssp.fut"


To meassure the performance of lssp, I generated the following dataset:
    $ futhark-dataset --i32-bounds=-50:50 -g [1000000]i32 > data

I then ran(multiple times) both the futhark-c compiled and the futhark-opencl compiled versions and get the following results:
    GPU (opencl)
    $ ./lssp -t /dev/stderr
    1720
    10i32

    CPU (c)
    2082
    10i32

Where the 10i32 is the result of lssp. 

We can see here that the running time of the opencl compiled version is faster than the cpu version. The speedup is not just a factor of adding the cores of the machine, which means that it is probably limited by the accesses to global memory. Had we increased the size of the array we would probably see a larger speedup in the GPU version.

