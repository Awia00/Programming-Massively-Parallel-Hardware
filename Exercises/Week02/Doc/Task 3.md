## Task II.2

Program:
    
    for(k=0; k<1024; k++) p += x[k]*y[k];

Vectorized

    for(k=0; k<1024; k+=64) {
        for(j=0; j<64; j++) // vectorize this inner loop
            p += x[k+j] * y[k+j];
    } //end of loop


### a)

        ADDI R6,R0,#64      // set R6 to the stride 64
        L.V V1,0(R1),R6     // load p
Loop:   L.V V2,0(R2),R6     // load slice of x 
        L.V V3,0(R3),R6     // load slice of y
        MUL.V V3,V1,V2      // x[k+j] * y[k+j] 
        ADD.V V1,V1,V3      // p += x[k+j] * y[k+j] 
        SUBBI R2,R2,#512    // jump to next slices,
        SUBBI R3,R3,#512    // jump to next slices,
        SUBBI R4,R4,R6      // jump by stride 
        BNEZ R4,LOOP        // 

I probably have misunderstood something here especially concerning the SUBBI instruction and why it jumps the 8*64.
Furthermore I am missing the strip-mined outer loop.


### b)

    startup(load/store)     = 30 cycles
    startup(mult)           = 10 cycles
    startup(add)            = 5 cycles
    Vector length           = 64
    Iterations              = 1024 / Vector Length = 16
    p += x[k+j] * y[k+j]    = 3 * startup(load) + startup(mult) + startup(addv) + startup(store) + Vector length <=> 3*30 + 10 + 5 + 30 + 64 = 199
        // not assuming loads in parallel - hence the 3 * startup(load)

    Now each of the 199 cycles must be done 16 times (for each iteration) totalling at: 3184 cycles.

### c)
Since the matrix matrix multiplication is just a dot product for each column and row, we can reuse the pervious result.
Foreach entry in the output matrix we have to calculate the dot product over some row and column. Since the output matrix also has 1024 entries we get: 
    
    1024 * 3184 = 3 260 416 cycles

Given enough space in the memory bank we can save one startup(load) dot product since each row must be multiplied with every column and therefore the load of a row from a previous iteration can be reused in the next iteration. Though we still need to pay the startup(load) once for each row. This results in the following number of cycles.

    169 * 16 * 1024 + 30*1024 = 2 799 616 cycles

