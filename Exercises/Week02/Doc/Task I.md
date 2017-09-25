## Task I.1

The main idea is to call scanInc or sgmScanInc and then shift the result to the right. 
It is tested the same way scanInc and sgmScanInc was tested in the provided code.

See CUDA folder for the implementation. To compile run "$ make" to run the program "$ make run"

I get the following results:

    Scan Exclusive GPU Kernel runs in: 46365 microsecs
    Scan Exclusive GPU Kernel runs in: 46376 microsecs
    Sgm Scan Exclusive GPU Kernel runs in: 63306 microsecs


## Task I.2

The code is based on the futhark implementation. First a map which makes the integers into MyInt4 and then a scan over the result with the mssp operator.
The implementation is tested on a large instance set(with random positive and negative integers) and is validated by comparison with a sequential imperative implementation. 
Note that the result of the implementation is the scanned array so the CPU code picks the last element for the reduction.

See CUDA folder for the implementation. To compile run "$ make" to run the program "$ make run"

I get the following results:

    MSSP GPU Kernel runs in:        103753 microsecs
    MSSP CPU runs in:               51466 microsecs

    MSSP GPU Kernel runs in:        103918 microsecs
    MSSP CPU runs in:               51082 microsecs

The difference is probably due to all the copying of memory and so on and the relatively low amount of aritmatic operations. 
... Or it could be due to the weird Nvidia update.


## Task I.3

Instead of flattening the sequential code directly, it was observed that matrix vector multiplication is a sgmScan with the plus operator. So the futhark implementation is based on creating the flags and then mapping the results of each row and then scanning over the result.
The cuda program is implemented almost as the futhark program, with the exceptions of pre-calculatation of the flag array, and the write_lastSgmElem method.

The implementation is tested on a decently sized square matrix with no sparse elements. It should work on matrices with zero entries but I ran out of time to implement a proper way of generating such a matrix. The implementation is validated with a sequential imperative implementation.

See CUDA and Futhark folders for the implementation. To compile run "$ make" to run the program "$ make run"

I get the following results:

    SP MV MUL GPU Kernel runs in:   297212 microsecs
    SP MV MUL CPU runs in:          161664 microsecs

    SP MV MUL GPU Kernel runs in:   296797 microsecs
    SP MV MUL CPU runs in:          161035 microsecs

The difference is probably due to all the copying of memory and the increased amount of work the flattened code has to do (increase is constant asymtotically). 
... Or it could be due to the weird Nvidia update.

When the matrix gets bigger the GPU implementation should clearly win over the CPU implementation. I had some weird bugs when I increased the size too much but i did not have the time to look more into it.

