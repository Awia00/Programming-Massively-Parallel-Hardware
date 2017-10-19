## Task 2

Could be based on code like this:
    lock
        x = x+1
    unlock

R1/X, W1/X, R2/X, W2/X, R3/X, W3/X, R4/X, W4/X

Notice a Read followed by a Write is equivalent to read exclusive (BusRdX)

### a) 

MESI without optimizations: 
	R1/X: Bus read request					-> P1 moves from I to E 	: 40 cycles
	W1/X: Process write						-> P1 moves from E to M 	: 1 cycle

	R2/X: Bus read request					-> P2 moves from I to S and sends P1 to S 	: 40 cycles
	W2/X: Bus upgrade						-> P2 moves from S to M, P1 moves to I 		: 10 cycles + 1 for the process write

    and so on...

That givesa the following number of clock cycles:
    (40 + 1) + 3(40+10+1) = 194 cycles

	(6+B) + 3(6+B + 10) = 182 bytes

### b)
MESI migratory optimized: 
    P1 is not affected since it does is only "Bus Read"
    P2 is bus read and bus upgrade which will be the indicator for the detection
    For P3 the protocal detects the bus read + bus upgrade and therefore changes hardware to make bus read exclusive on bus read. 
        It does so by delaying the bus read transaction until it sees the bus upgrade and the combines them to Bus Read Exclusive.
    The same happens for P4

    p1: 40          Bus read 
    p2: 40 + 10     Bus read + bus upgrade
    p3: 40          Bus read exclusive
    p4: 40          Bus read exclusive

    170 cycles:

    p1: 6+B         Bus read 
    p2: 6+B + 10    Bus read + bus upgrade
    p3: 6+B         Bus read exclusive
    p4: 6+B         Bus read exclusive

	3(6+32) + (6+32+10) = 162 bytes.
	
Compare this to the 182 bytes of the MESI protocol without migratory optimization. It is an improvement of ~11%

