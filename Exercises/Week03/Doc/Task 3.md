## Task 3

### Task a,c,d)
See CUDA folder for implementation.

M: 4000 N: 5000 U: 3000

Matrix Matrix Mul naive
    MMM naive time           504636
    MMM gigaFlops naive      237.795175'
Matrix Matrix Mul optimized
    MMM optimized time       238004
    MMM gigaFlops optimized  504.193231

Nice ~2 speedup for the optimized version showing how the shared memory can speed up. 
I do not have a sequential version to compare speed with (since it takes a looooong time to complete), but I tried to run it on smaller instances for validation and it completed successfully. 

### Task b)
See the cuda folder for implementation (main.cu)
Compile with 'make omp'
Run with 'make runOMP'

M: 2200 N: 2000 U: 1800
Running time:
    Matrix Matrix Mul OMP time               2536824
    Matrix Matrix Mul sequential time        25878546

A nice ^10 time speedup - given the low dimension lengths this seems reasonable but I did not try on larger instances since it took too long.