# Exercises Week 2
*Anders Wind Steffensen* (qrj742@alumni.ku.dk)
25-09-2017

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

## Task II.1

The program:

    SEARCH:     LW R5, 0(R3)        /I1 Load item
                SUB R6, R5, R2      /I2 compare with key
                BNEZ R6, NOMATCH    /I3 check for match
                ADDI R1, R1, #1     /I4 count matches
    NOMATCH:    ADDI R3, R3, #4     /I5 next item
                BNE R4, R3, SEARCH  /I6 continue until all items



### a)
    SEARCH:     LW R5, 0(R3)        /I1 Load item
                NOOP
                NOOP
                NOOP
                SUB R6, R5, R2      /I2 compare with key
                NOOP
                NOOP
                NOOP
                BNEZ R6, NOMATCH    /I3 check for match
                ADDI R1, R1, #1     /I4 count matches
    NOMATCH:    ADDI R3, R3, #4     /I5 next item
                NOOP
                NOOP
                NOOP
                BNE R4, R3, SEARCH  /I6 continue until all items

l2 is delayed since it is a RAW on R5 from l1. Therefore the ID step of l2 must be after the WB step of l1 (3 clocks stall).
l3 is delayed since it is a RAW on R6 from l2. Therefore the ID step of l3 must be after the WB step of l2 (3 clocks stall).
l6 is delayed since it is a RAW on R4 from l5 (also R3 from l4 but l5 is the most binding). Therefore the ID step of l6 must be after the WB step of l5 (3 clocks stall).

### b)
Stalling on Data hazards.

#### Match
Assuming a match the pipeline looks like this.

|    | Clock              | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | C11 | C12 | C13 | C14 | C15 | C16 | C17 | C18 | C19 |
|----|--------------------|----|----|----|----|----|----|----|----|----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| l1 | LW R5, 0(R3)       | IF | ID | EX | ME | WB |    |    |    |    |     |     |     |     |     |     |     |     |     |     |
| l2 | SUB R6, R5, R2     |    | IF | ID | ID | ID | ID | EX | ME | WB |     |     |     |     |     |     |     |     |     |     |
| l3 | BNEZ R6, NOMATCH   |    |    | IF | IF | IF | IF | ID | ID | ID | ID  | EX  | ME  | WB  |     |     |     |     |     |     |
| l4 | ADDI R1, R1, #1    |    |    |    |    |    |    | IF | IF | IF | IF  | ID  | EX  | ME  | WB  |     |     |     |     |     |
| l5 | ADDI R3, R3, #4    |    |    |    |    |    |    |    |    |    |     | IF  | ID  | EX  | ME  | WB  |     |     |     |     |
| l6 | BNE R4, R3, SEARCH |    |    |    |    |    |    |    |    |    |     |     | IF  | ID  | ID  | ID  | ID  | EX  | ME  | WB  |

As can be seen in the table the program would finish in clock 19.

#### Non match
Assuming a non-match the pipeline looks like this

|    | Clock              | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | C11 | C12 | C13 | C14 | C15 | C16 | C17 | C18 | C19 | C20 |
|----|--------------------|----|----|----|----|----|----|----|----|----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| l1 | LW R5, 0(R3)       | IF | ID | EX | ME | WB |    |    |    |    |     |     |     |     |     |     |     |     |     |     |     |
| l2 | SUB R6, R5, R2     |    | IF | ID | ID | ID | ID | EX | ME | WB |     |     |     |     |     |     |     |     |     |     |     |
| l3 | BNEZ R6, NOMATCH   |    |    | IF | IF | IF | IF | ID | ID | ID | ID  | EX  | ME  | WB  |     |     |     |     |     |     |     |
| l4 | ADDI R1, R1, #1    |    |    |    |    |    |    | IF | IF | IF | IF  | ID  |     |     |     |     |     |     |     |     |     |
| l5 | ADDI R3, R3, #4    |    |    |    |    |    |    |    |    |    |     | IF  | IF  | ID  | EX  | ME  | WB  |     |     |     |     |
| l6 | BNE R4, R3, SEARCH |    |    |    |    |    |    |    |    |    |     |     |     | IF  | ID  | ID  | ID  | ID  | EX  | ME  | WB  |

At C12 the IF and ID stages are flushed since the branch is taken.
Therefore the program would finish in clock 20



### c)
With register forwarding.

#### Match
Assuming a match the pipeline looks like this.

|    | Clock              | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | C11 | C12 | C13 | C14 | C15 | C16 |
|----|--------------------|----|----|----|----|----|----|----|----|----|-----|-----|-----|-----|-----|-----|-----|
| l1 | LW R5, 0(R3)       | IF | ID | EX | ME | WB |    |    |    |    |     |     |     |     |     |     |     |
| l2 | SUB R6, R5, R2     |    | IF | ID | ID | ID | EX | ME | WB |    |     |     |     |     |     |     |     |
| l3 | BNEZ R6, NOMATCH   |    |    | IF | IF | IF | ID | ID | ID | EX | ME  | WB  |     |     |     |     |     |
| l4 | ADDI R1, R1, #1    |    |    |    |    |    | IF | IF | IF | ID | EX  | ME  | WB  |     |     |     |     |
| l5 | ADDI R3, R3, #4    |    |    |    |    |    |    |    |    | IF | ID  | EX  | ME  | WB  |     |     |     |
| l6 | BNE R4, R3, SEARCH |    |    |    |    |    |    |    |    |    | IF  | ID  | ID  | ID  | EX  | ME  | WB  |

As can be seen in the table the program would finish in clock 16 effectively decreasing the # of clocks by 3 which makes good sense since we saved 1 clock for each RAW hazard.

#### Non match
Assuming a non-match the pipeline looks like this

|    | Clock              | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | C11 | C12 | C13 | C14 | C15 | C16 | C17 |
|----|--------------------|----|----|----|----|----|----|----|----|----|-----|-----|-----|-----|-----|-----|-----|-----|
| l1 | LW R5, 0(R3)       | IF | ID | EX | ME | WB |    |    |    |    |     |     |     |     |     |     |     |     |
| l2 | SUB R6, R5, R2     |    | IF | ID | ID | ID | EX | ME | WB |    |     |     |     |     |     |     |     |     |
| l3 | BNEZ R6, NOMATCH   |    |    | IF | IF | IF | ID | ID | ID | EX | ME  | WB  |     |     |     |     |     |     |
| l4 | ADDI R1, R1, #1    |    |    |    |    |    | IF | IF | IF | ID |     |     |     |     |     |     |     |     |
| l5 | ADDI R3, R3, #4    |    |    |    |    |    |    |    |    | IF | IF  | ID  | EX  | ME  | WB  |     |     |     |
| l6 | BNE R4, R3, SEARCH |    |    |    |    |    |    |    |    |    |     | IF  | STL | STL | ID  | EX  | ME  | WB  |

Therefore the program would finish in clock 17, also decreasing the number of clocks with 3.

### d)
With full forwarding

#### Match
Assuming a match the pipeline looks like this.

|    | Clock              | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | C11 | C12 | C13 |
|----|--------------------|----|----|----|----|----|----|----|----|----|-----|-----|-----|-----|
| l1 | LW R5, 0(R3)       | IF | ID | EX | ME | WB |    |    |    |    |     |     |     |     |
| l2 | SUB R6, R5, R2     |    | IF | ID | ID | EX | ME | WB |    |    |     |     |     |     |
| l3 | BNEZ R6, NOMATCH   |    |    | IF | IF | ID | ID | EX | ME | WB |     |     |     |     |
| l4 | ADDI R1, R1, #1    |    |    |    |    | IF | IF | ID | EX | WB |     |     |     |     |
| l5 | ADDI R3, R3, #4    |    |    |    |    |    |    | IF | ID | EX | ME  | WB  |     |     |
| l6 | BNE R4, R3, SEARCH |    |    |    |    |    |    |    | IF | ID | ID  | EX  | ME  | WB  |

As can be seen in the table the program would finish in clock 13 effectively decreasing the # of clocks by 3.

#### Non match
Assuming a non-match the pipeline looks like this

|    | Clock              | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | C11 | C12 | C13 | C14 |
|----|--------------------|----|----|----|----|----|----|----|----|----|-----|-----|-----|-----|-----|
| l1 | LW R5, 0(R3)       | IF | ID | EX | ME | WB |    |    |    |    |     |     |     |     |     |
| l2 | SUB R6, R5, R2     |    | IF | ID | ID | EX | ME | WB |    |    |     |     |     |     |     |
| l3 | BNEZ R6, NOMATCH   |    |    | IF | IF | ID | ID | EX | ME | WB |     |     |     |     |     |
| l4 | ADDI R1, R1, #1    |    |    |    |    | IF | IF | ID |    |    |     |     |     |     |     |
| l5 | ADDI R3, R3, #4    |    |    |    |    |    |    | IF | IF | ID | EX  | ME  | WB  |     |     |
| l6 | BNE R4, R3, SEARCH |    |    |    |    |    |    |    |    | IF | ID  | ID  | EX  | ME  | WB  |

Therefore the program would finish in clock 14, also decreasing the number of clocks with 3.


### e)
I showcase a single unroll but it should generalize well to more than one unroll. 
Assuming that there is an even number of items in the list, I change the program in the following way:

    SEARCH:     LW R5, 0(R3)        /I1 Load item1
                LW R8, 4(R7)        /I2 Load item2
                SUB R6, R5, R2      /I3 compare with key
                SUB R9, R8, R2      /I4 compare with key
                BNEZ R6, NOMATCH1   /I5 check for match
                ADDI R1, R1, #1     /I6 count matches    
    NOMATCH1:   BNEZ R9, NOMATCH2   /I7 check for match
                ADDI R1, R1, #1     /I8 count matches
    NOMATCH2:   ADDI R3, R3, #8     /I9 next item
                ADDI R7, R7, #8     /I10 next item
                BNE R4, R7, SEARCH  /I11 continue until all items

|     | Clock              | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | C11 | C12 | C13 | C14 | C14 |
|-----|--------------------|----|----|----|----|----|----|----|----|----|-----|-----|-----|-----|-----|-----|
| l1  | LW R5, 0(R3)       | IF | ID | EX | ME | WB |    |    |    |    |     |     |     |     |     |     |
| l2  | LW R8, 0(R7)       |    | IF | ID | EX | ME | WB |    |    |    |     |     |     |     |     |     |
| l3  | SUB R6, R5, R2     |    |    | IF | ID | EX | ME | WB |    |    |     |     |     |     |     |     |
| l4  | SUB R9, R8, R2     |    |    |    | IF | ID | EX | ME | WB |    |     |     |     |     |     |     |
| l5  | BNEZ R6, NOMATCH1  |    |    |    |    | IF | ID | EX | ME | WB |     |     |     |     |     |     |
| l6  | ADDI R1, R1, #1    |    |    |    |    |    | IF | ID | EX | ME | WB  |     |     |     |     |     |
| l7  | BNEZ R9, NOMATCH2  |    |    |    |    |    |    | IF | ID | EX | ME  | WB  |     |     |     |     |
| l8  | ADDI R1, R1, #1    |    |    |    |    |    |    |    | IF | ID | EX  | ME  | WB  |     |     |     |
| l9  | ADDI R3, R3, #4    |    |    |    |    |    |    |    |    | IF | ID  | EX  | ME  | WB  |     |     |
| l10 | ADDI R7, R7, #4    |    |    |    |    |    |    |    |    |    | IF  | ID  | EX  | ME  | WB  |     |
| l11 | BNE R4, R7, SEARCH |    |    |    |    |    |    |    |    |    |     | IF  | ID  | EX  | ME  | WB  |

At only 1 more clock than the unrolled version, 2 iterations of the loop can be calculated. Futhermore this is optimal since there is no stalls.

In the case where a branch is taken stalls still occur as shown on the table below.

|     | Clock              | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | C11 | C12 | C13 | C14 | C14 | C15 | C15 |
|-----|--------------------|----|----|----|----|----|----|----|----|----|-----|-----|-----|-----|-----|-----|-----|-----|
| l1  | LW R5, 0(R3)       | IF | ID | EX | ME | WB |    |    |    |    |     |     |     |     |     |     |     |     |
| l2  | LW R8, 0(R7)       |    | IF | ID | EX | ME | WB |    |    |    |     |     |     |     |     |     |     |     |
| l3  | SUB R6, R5, R2     |    |    | IF | ID | EX | ME | WB |    |    |     |     |     |     |     |     |     |     |
| l4  | SUB R9, R8, R2     |    |    |    | IF | ID | EX | ME | WB |    |     |     |     |     |     |     |     |     |
| l5  | BNEZ R6, NOMATCH1  |    |    |    |    | IF | ID | EX | ME | WB |     |     |     |     |     |     |     |     |
| l6  | ADDI R1, R1, #1    |    |    |    |    |    | IF | ID |    |    |     |     |     |     |     |     |     |     |
| l7  | BNEZ R9, NOMATCH2  |    |    |    |    |    |    | IF | IF | ID | EX  | ME  | WB  |     |     |     |     |     |
| l8  | ADDI R1, R1, #1    |    |    |    |    |    |    |    |    | IF | ID  |     |     |     |     |     |     |     |
| l9  | ADDI R3, R3, #4    |    |    |    |    |    |    |    |    |    | IF  | IF  | ID  | EX  | ME  | WB  |     |     |
| l10 | ADDI R7, R7, #4    |    |    |    |    |    |    |    |    |    |     | IF  | ID  | EX  | ME  | WB  |     |     |
| l11 | BNE R4, R7, SEARCH |    |    |    |    |    |    |    |    |    |     |     | IF  | ID  | ID  | EX  | ME  | WB  |

Here we still have two stalls at l7 and l9. Delaying branches could help by placing the L9 and L10 instructions right after the branch operation since R3 and R7 is not used by any later operations.

    SEARCH:     LW R5, 0(R3)        /I1 Load item1
                LW R8, 4(R7)        /I2 Load item2
                SUB R6, R5, R2      /I3 compare with key
                SUB R9, R8, R2      /I4 compare with key
                BNEZ R6, NOMATCH1   /I5 check for match
    (delayslot) ADDI R3, R3, #8     /l6 delay slot next item   
                ADDI R1, R1, #1     /I7 count matches    
    NOMATCH1:   BNEZ R9, NOMATCH2   /I8 check for match
    (delayslot) ADDI R7, R7, #8     /I9 delay slot next item
                ADDI R1, R1, #1     /I10 count matches
    NOMATCH2:   BNE R4, R7, SEARCH  /I11 continue until all items

Results in the following:

|     | Clock              | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | C11 | C12 | C13 | C14 | C14 |
|-----|--------------------|----|----|----|----|----|----|----|----|----|-----|-----|-----|-----|-----|-----|
| l1  | LW R5, 0(R3)       | IF | ID | EX | ME | WB |    |    |    |    |     |     |     |     |     |     |
| l2  | LW R8, 0(R7)       |    | IF | ID | EX | ME | WB |    |    |    |     |     |     |     |     |     |
| l3  | SUB R6, R5, R2     |    |    | IF | ID | EX | ME | WB |    |    |     |     |     |     |     |     |
| l4  | SUB R9, R8, R2     |    |    |    | IF | ID | EX | ME | WB |    |     |     |     |     |     |     |
| l5  | BNEZ R6, NOMATCH1  |    |    |    |    | IF | ID | EX | ME | WB |     |     |     |     |     |     |
| l9  | ADDI R3, R3, #4    |    |    |    |    |    | IF | ID | EX | ME | WB  |     |     |     |     |     |
| l6  | ADDI R1, R1, #1    |    |    |    |    |    |    | IF |    |    |     |     |     |     |     |     |
| l7  | BNEZ R9, NOMATCH2  |    |    |    |    |    |    |    | IF | ID | EX  | ME  | WB  |     |     |     |
| l10 | ADDI R7, R7, #4    |    |    |    |    |    |    |    |    | IF | ID  | EX  | ME  | WB  |     |     |
| l8  | ADDI R1, R1, #1    |    |    |    |    |    |    |    |    |    | IF  |     |     |     |     |     |
| l11 | BNE R4, R7, SEARCH |    |    |    |    |    |    |    |    |    |     | IF  | ID  | EX  | ME  | WB  |

Ending up in the same amount of instruction as for the no matches case. We know this is optimal since we have no stalls. 
This program would also be optimal with no matches.

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

                                // at some point a startup(store) for storing p

    Now each of the 109 cycles must be done 16 times (for each iteration) and adding 30 for startup(store) totalling at: 1744+30 = 1774 cycles.

### c)
Since the matrix matrix multiplication is just a dot product for each column and row, we can reuse the previous result.
Foreach entry in the output matrix we have to calculate the dot product over some row and column. Since the output matrix also has 1024*1024 entries we get: 
    
    1024 * 1024 * 1774 cycles = 1 860 173 824 cycles
    
