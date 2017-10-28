## Task 1

### a,c,d)
See cuda folder for implementation

8000*7000 matrix

Running time:
    M-transpose naive time                   87316
    M-transpose optimized time               90968
    Matrix Transpose sequential time         1015279

The optimized did not do better than the unoptimized. This might have improved if the dimensions were higher. 
We do see a ~10-11 speed up from CPU to GPU version though.

### b)
See cuda folder for implementation (main.cu)
Compile with 'make omp'
Run with 'make runOMP'

6000*5000 matrix:

Running time:
    Matrix Transpose OMP time                74615
    Matrix Transpose sequential time         559462

Close to a factor ~7 speedup. Given that the server has 32 cores but only one of the loops was parallelized this is a nice speedup.