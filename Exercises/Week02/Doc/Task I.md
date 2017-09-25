## Task I.1

The main idea is to call scanInc and sgmScanInc and then shift the result to the right. 
It is tested the same way scanInc and sgmScanInc was tested in the provided code.

See CUDA folder for the implementation. To compile run "$ make" to run the program "$ make run"

## Task I.2

The code is based on the futhark implementation. First a map which makes the integers into MyInt4 and then a scan over the result with the mssp operator.
The implementation is tested on a large instance set(with random positive and negative numbers) and is validated by comparison with a sequential imperative implementation. 
Note that the result of the implementation is the scanned array so the CPU code picks the last element for the reduction.

See CUDA folder for the implementation. To compile run "$ make" to run the program "$ make run"

## Task I.3

Instead of flattening the sequential code directly, it was observed that matrix vector multiplication is a sgmScan with the plus operator. So the futhark implementation is based on creating the flags and then mapping the results of each row and then scanning over the result.
The cuda program is implemented almost as the futhark program, with the exceptions of pre-calculatation of the flag array, and the write_lastSgmElem method.

The implementation is tested on a decently sized square matrix with no sparse elements. It should work on matrices with zero entries but I ran out of time to make a proper way of generating such a matrix. The implementation is validated with a sequential imperative implementation.

See CUDA and Futhark folders for the implementation. To compile run "$ make" to run the program "$ make run"

