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
I showcase a single unroll. I change the program in the following way. I assume that there is an even number of items in the list.

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

At only 1 more clock, 2 iterations of the loop can be calculated.
In the case where a branch is taken we have another situation.

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

Ending up in the same amount of instruction as for the no matches case. We know this is optimal since we have no stalls.