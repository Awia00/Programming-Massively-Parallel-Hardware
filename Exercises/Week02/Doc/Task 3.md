## Task II.2

Program:
    
    for(k=0; k<1024; k++) p += x[k]*y[k];

Vectorized

    for(k=0; k<1024; k+=64) {
        for(j=0; j<64; j++) // vectorize this inner loop
            p += x[k+j] * y[k+j];
    } //end of loop


### a)

I take a look at the inner loop

        R1 holds the stride = 1
        R2 holds current address of x[k]
        R3 holds current address of x[k]
        V4 holds vector p
        
        // since the inner loop is 64 length we dont need to write it as a loop since it all fits in vector
        L.V V1, 0(R2) R1    // load x[k] and the 64 entries 
        L.V V2, 0(R3) R1    // load y[k] and the 64 entries 
        MUL.V   V3,V1,V2    // x[k+j] * y[k+j] 
        ADD.V   V4,V4,V3    // p = x[k+j] * y[k+j]

### b)

    startup(load/store)     = 30 cycles
    startup(mult)           = 10 cycles
    startup(add)            = 5 cycles
    Vector length           = 64
    Iterations              = 1024 / Vector Length = 16
    p += x[k+j] * y[k+j]    = 
                                L.V V1, 0(R2) R1: startup(load)
                                L.V V2, 0(R3) R1: startup(load)
                                MUL.V   V3,V1,V2: startup(mul)
                                ADD.V   V4,V4,V3: startup(add)
                                
                                30 + 10 + 5 + 64 = 109

    Now each of the 109 cycles must be done 16 times (for each iteration) totalling at: 1744 = 1744 cycles.

### c)
Since the matrix matrix multiplication is just a dot product for each column and row, we can reuse the previous result.
Foreach entry in the output matrix we have to calculate the dot product over some row and column. Since the output matrix also has 1024*1024 entries we get: 
    
    1024 * 1024 * 1744 cycles = 1 828 716 544 cycles
    
