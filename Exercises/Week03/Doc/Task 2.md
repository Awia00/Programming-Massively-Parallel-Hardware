## Task 2

### a)
The loop is not parallel since it is dependend on *accum* which is changed in every iteration of the loop based on the value from the i-iteration before.
To make the outloop parallel one could use the concept of privatization. That is either create accum inside the loop, make it an array and index into it with the loop variable.
The same problem applies to tmpA which is also reused in every iteration.

1. 	for i from 0 to N-1 // outer loop
2. 		accum[i] = A[i,0] * A[i,0];
3. 		B[i,0] = accum[i];
4. 		for j from 1 to 63 // inner loop
5. 			tmpA[i] = A[i, j];
6. 			accum[i] = sqrt(accum[i]) + tmpA[i]*tmpA[i];
7. 			B[i,j] = accum[i];

or accum and tmpA can be removed all together like this

1. 	for i from 0 to N-1 // outer loop
3. 		B[i,0] = A[i,0] * A[i,0];
4. 		for j from 1 to 63 // inner loop
7. 			B[i,j] = sqrt(B[i,j-1]) + A[i, j]*A[i, j];

The inner loop is not map parallel since each iteration is dependend on the resulting *accum* from the j-iteration before.

The innerloop cannot be written as a composition of parallel operators. The plus binary operator is associative, but when applying *sqrt* to elements which are accumulated over iterations then the result cannot be calculated out of order. For example: sqrt(sqrt(a) + b) + c is not the same as sqrt(sqrt(b) + a) + c, which represents an out of order execution. 

When sqrt is removed from the expression, all the operators(both + and *) are associative and therefore it can be written as a composition of parallel operators, namely a map squaring the A[i,j] and then a scan sum.

### b) 
See the cuda folder for implementation (main.cu)
Compile with 'make omp'
Run with 'make runOMP'

64*200000 matrix

Running time :
    Square Accumulator OMP time              7419
    Square Accumulator sequential time       141303

So a nice factor ~20 speedup. Which given that the server has 32 cores is quite nice and realistic. Also note that the dimension which was parallelized was much larger than the inner loop (compared to the 7~ speedup of matrix transpose OMP where both dimensions had almost equal values)

### c,d) 
See cuda folder for implementation

64*400000 matrix

Running time:
    Square Accumulator naive time            17119
    Square Accumulator optimized time        3767
    Square Accumulator sequential time       360228

Here we are seeing some wery nice results showing how much parallelism we get. the optimized version is close to ~100 times faster than the CPU version and ~5 speedup from the unoptimized non-transposed version, even with the added extra computation of 2x transpose.

