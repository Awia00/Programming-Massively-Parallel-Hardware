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
#### Match
Assuming a match the pipeline looks like this.

|    | Clock              | C1 | C2   | C3   | C4   | C5 | C6   | C7   | C8   | C9 | C10 | C11 | C12  | C13  | C14  | C15 | C16 | C17 | C18 | C19 |
|----|--------------------|----|------|------|------|----|------|------|------|----|-----|-----|------|------|------|-----|-----|-----|-----|-----|
| l1 | LW R5, 0(R3)       | IF | ID   | EX   | ME   | WB |      |      |      |    |     |     |      |      |      |     |     |     |     |     |
| l2 | SUB R6, R5, R2     |    | NOOP | NOOP | NOOP | IF | ID   | EX   | ME   | WB |     |     |      |      |      |     |     |     |     |     |
| l3 | BNEZ R6, NOMATCH   |    |      |      |      |    | NOOP | NOOP | NOOP | IF | ID  | EX  | ME   | WB   |      |     |     |     |     |     |
| l4 | ADDI R1, R1, #1    |    |      |      |      |    |      |      |      |    | IF  | ID  | EX   | ME   | WB   |     |     |     |     |     |
| l5 | ADDI R3, R3, #4    |    |      |      |      |    |      |      |      |    |     | IF  | ID   | EX   | ME   | WB  |     |     |     |     |
| l6 | BNE R4, R3, SEARCH |    |      |      |      |    |      |      |      |    |     |     | NOOP | NOOP | NOOP | IF  | ID  | EX  | ME  | WB  |

|    | Clock              | C1 | C2   | C3  | C4  | C5  | C6  | C7  | C8  | C9  | C10 | C11 | C12 | C13 | C14 | C15 | C16 | C17 | C18 | C19 |
|----|--------------------|----|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| l1 | LW R5, 0(R3)       | IF | ID   | EX  | ME  | WB  |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
| l2 | SUB R6, R5, R2     |    | IF   | STL | STL | STL | ID  | EX  | ME  | WB  |     |     |     |     |     |     |     |     |     |     |
| l3 | BNEZ R6, NOMATCH   |    |      | IF  | STL | STL | STL | STL | STL | STL | ID  | EX  | ME  | WB  |     |     |     |     |     |     |
| l4 | ADDI R1, R1, #1    |    |      |     | IF  | STL | STL | STL | STL | STL | STL | ID  | EX  | ME  | WB  |     |     |     |     |     |
| l5 | ADDI R3, R3, #4    |    |      |     |     | IF  | STL | STL | STL | STL | STL | STL | ID  | EX  | ME  | WB  |     |     |     |     |
| l6 | BNE R4, R3, SEARCH |    |      |     |     |     | IF  | STL | STL | STL | STL | STL | STL | STL | STL | STL | ID  | EX  | ME  | WB  |

As can be seen in the table the program would finish in clock 19.

#### Non match
Assuming a non-match the pipeline looks like this

|    | Clock              | C1 | C2   | C3  | C4  | C5  | C6  | C7  | C8  | C9  | C10 | C11 | C12 | C13 | C14 | C15 | C16 | C17 | C18 | C19 | C20 |
|----|--------------------|----|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| l1 | LW R5, 0(R3)       | IF | ID   | EX  | ME  | WB  |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
| l2 | SUB R6, R5, R2     |    | IF   | STL | STL | STL | ID  | EX  | ME  | WB  |     |     |     |     |     |     |     |     |     |     |     |
| l3 | BNEZ R6, NOMATCH   |    |      | IF  | STL | STL | STL | STL | STL | STL | ID  | EX  | ME  | WB  |     |     |     |     |     |     |     |
| l4 | ADDI R1, R1, #1    |    |      |     | IF  | STL | STL | STL | STL | STL | STL | ID  |     |     |     |     |     |     |     |     |     |
| l5 | ADDI R3, R3, #4    |    |      |     |     | IF  | STL | STL | STL | STL | STL | STL | IF  | ID  | EX  | ME  | WB  |     |     |     |     |
| l6 | BNE R4, R3, SEARCH |    |      |     |     |     | IF  | STL | STL | STL | STL | STL | STL | IF  | STL | STL | STL | ID  | EX  | ME  | WB  |

|    | Clock              | C1 | C2   | C3   | C4   | C5 | C6   | C7   | C8   | C9 | C10 | C11 | C12  | C13  | C14  | C15  | C16 | C17 | C18 | C19 | C20 |
|----|--------------------|----|------|------|------|----|------|------|------|----|-----|-----|------|------|------|------|-----|-----|-----|-----|-----|
| l1 | LW R5, 0(R3)       | IF | ID   | EX   | ME   | WB |      |      |      |    |     |     |      |      |      |      |     |     |     |     |     |
| l2 | SUB R6, R5, R2     |    | NOOP | NOOP | NOOP | IF | ID   | EX   | ME   | WB |     |     |      |      |      |      |     |     |     |     |     |
| l3 | BNEZ R6, NOMATCH   |    |      |      |      |    | NOOP | NOOP | NOOP | IF | ID  | EX  | ME   | WB   |      |      |     |     |     |     |     |
| l4 | ADDI R1, R1, #1    |    |      |      |      |    |      |      |      |    | IF  | ID  |      |      |      |      |     |     |     |     |     |
| l5 | ADDI R3, R3, #4    |    |      |      |      |    |      |      |      |    |     | IF  | IF   | ID   | EX   | ME   | WB  |     |     |     |     |
| l6 | BNE R4, R3, SEARCH |    |      |      |      |    |      |      |      |    |     |     |      | NOOP | NOOP | NOOP | IF  | ID  | EX  | ME  | WB  |

At C12 the IF and ID stages are flushed since the branch is taken.
Therefore the program would finish in clock 20



### c)


|    | Clock              | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 |
|----|--------------------|----|----|----|----|----|----|----|----|----|-----|
| l1 | LW R5, 0(R3)       | IF | ID | EX | ME | WB |    |    |    |    |     |
| l2 | SUB R6, R5, R2     |    | IF | ID | EX | ME | WB |    |    |    |     |
| l3 | BNEZ R6, NOMATCH   |    |    | IF | ID | EX | ME | WB |    |    |     |
| l4 | ADDI R1, R1, #1    |    |    |    | IF | ID | EX | ME | WB |    |     |
| l5 | ADDI R3, R3, #4    |    |    |    |    | IF | ID | EX | ME | WB |     |
| l6 | BNE R4, R3, SEARCH |    |    |    |    |    | IF | ID | EX | ME | WB  |