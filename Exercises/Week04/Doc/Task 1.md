## Task 1
### a)

R1/X, W1/X, W1/X, R2/X, W2/X, W2/X, R3/X, W3/X, W3/X, R4/X, W4/X, W4/X 


*MSI:* 
MSI has three states Modified, Shared and Invalid.
    M <-> S <-> I

	We get the following operations, transactions and clock cycles:

    R1/X: Bus read request (with read miss)	-> P1 moves from I to S : 40 cycles
    W1/X: Bus upgrade						-> P1 moves from S to M : 10 cycles (+ 1 cycle for wite hit)
    W1/X: Write hit							-> P1 stays in M        : 1 cycle (write hit)

    R2/X: Bus read request (with read miss)	-> P2 moves from I to S and P1 moves from M to S    : 40 cycles
    W2/X: Bus upgrade						-> P2 moves from S to M, P1 moves to invalid        : 10 cycles (+ 1 cycle write hit)
    W2/X: Write hit							-> P2 stay in M                                     : 1 cycle

And so on for the last two processes. To calculate the number of cycles we split look at the cost for each process.
	(40 + 10 + 1 + 1)
Each process usews equally many cycles therefore:
	(40 + 10 + 1 + 1)*4 = 208.

Cosmin did not include the write hits but said we could. If no included the total number of clock cycles are 200.


*MESI:*
The MESI protocol has 4 states, Modified, Exclusive, Shared and Invalid.
	M <-> E <-> S <-> I

	We get the following operations, transactions and clock cycles:

	R1/X: Bus read request					-> P1 moves from I to E 	: 40 cycles
	W1/X: Process write						-> P1 moves from E to M 	: 1 cycle 
	W1/X: Process write						-> P1 stays in M      		: 1 cycle

	R2/X: Bus read request					-> P2 moves from I to S and sends P1 to S 	: 40 cycles
	W2/X: Bus upgrade						-> P2 moves from S to M, P1 moves to I 		: 10 cycles + 1 for the process write
	W2/X: Process write						-> P2 stays in M							: 1 cycle

And so on for the last two processes. To calculate the number of cycles we split look at the cost for each process.
	P1: 	40+ 1+1
	P2-P4:  40+10+1
Therefore we get:
	42+3(40+10+1+1) = 198.

Therefore the MESI protocol does better in this case.

### b)
Assuming B = 32 bytes.

MSI:
    4*(read request + bus upgrade)
    4*(6 + 32 + 10) = 192 bytes

MESI
	P1: 	read request
	P2-P4	Read request + bus upgrade
    (6 + 32) + 3(6 + 32 + 10) = 182 bytes

Again MESI comes out on top.

