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
    Square Accumulator naive time            4848
    Square Accumulator optimized time        19

But it does not seem precise as the running time feels very similar.

