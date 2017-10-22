# Exercises Week 3 - Resubmit
*Anders Wind Steffensen* (qrj742@alumni.ku.dk)
19-10-2017

## Task 1

See cuda folder for implementation

5000*5000 matrix

Running time:
    M-transpose naive time           55
    M-transpose optimized time       59

Which weirdly enough is very close to eachother. Some results also indicate a faster optimized version but in average they are very equal. This is probably due to some implementation error or the GPU01 nvidia driver.

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

### c,d) 
See cuda folder for implementation

64*250000 matrix

Running times:
    Square Accumulator naive time            13015
    Square Accumulator optimized time        7818

We are now seeing a nice speedup of transposing matrix A and then transposing matrix B to get the wanted result.

## Task 3

See CUDA folder for implementation.

8000*8000 matrices.

Running Times:
    MMM naive time           11
    MMM optimized time       12

Gigaflops:
    MMM gigaFlops naive      93090913.512490
    MMM gigaFlops optimized  85333337.386449

Again naive is faster than the optimized. Probably due to a fault in the implementation.

