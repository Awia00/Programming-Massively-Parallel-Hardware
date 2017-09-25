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

l2 is delayed since it is a RAW on R5 from l1. Therefore the ID step of l2 must be after the WB step of l1.
l3 is delayed since it is a RAW on R6 from l2. Therefore the ID step of l3 must be after the WB step of l2.
l6 is delayed since it is a RAW on R4 from l5 (also R3 from l4 but l5 is the most binding). Therefore the ID step of l6 must be after the WB step of l5.


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
I showcase a single unroll but it should generalize well to more than 1 unroll. I change the program in the following way. I assume that there is an even number of items in the list.

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

At only 1 more clock than the unrolled version, 2 iterations of the loop can be calculated.

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

Here we still have two stall at l7 and l9. Delaying branches could help by placing the L9 and L10 instructions right after the branch since R3 and R7 is not used by any following instructions.

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

Ending up in the same amount of instruction as for the no matches case. We know this is optimal since we have no stalls. This program is also just as fast with no matches.

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

I probably have misunderstood something here. Concerning the SUBBI instruction and why it jumps the 8*64.
Furthermore I am missing the strip-mined loop.


### b)

    startup(load/store)     = 30 cycles
    startup(mult)           = 10 cycles
    startup(add)            = 5 cycles
    Vector length           = 64
        // this is I have assumed is the hardware vector size.
    Iterations              = 1024 / Vector Length = 16
    p += x[k+j] * y[k+j]    = 3 * startup(load) + startup(mult) + startup(addv) + startup(store) + Vector length <=> 3*30 + 10 + 5 + 30 + 64 = 199
        // not assuming loads in parallel - hence the 3 * startup(load)

    Now each of the 199 cycles must be done 16 times (for each iteration) totalling at: 3184 cycles.

### c)
Since the matrix matrix multiplication is just a dot product for each column and row, we can reuse the pervious result.
Foreach entry in the output matrix we have to calculate the dot product over some row and column. Since the output matrix also has 1024 entries we get: 
    
    1024 * 3184 cycles = 3 260 416 cycles

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

Instead of flattening the sequential code, it was observed that matrix vector multiplication is a sgmScan with the plus operator. So the futhark implementation is based on creating the flags and then mapping the results of each row and then scanning over the result.
The cuda program is implemented almost as the futhark program, with the exception that I choose to just calculate the flag array in advance. 
The implementation is tested on a decently sized square matrix with no sparse elements. It should work on matrices with zero entries but I ran out of time to make a proper way of generating such a matrix. The implementation is validated with a sequential imperative implementation.

See CUDA and Futhark folders for the implementation. To compile run "$ make" to run the program "$ make run"

# Exercises Week 1
*Anders Wind Steffensen* (qrj742@alumni.ku.dk)
25-09-2017
