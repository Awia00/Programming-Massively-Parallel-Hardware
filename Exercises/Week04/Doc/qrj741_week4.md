# Exercises Week 3
*Anders Wind Steffensen* (qrj742@alumni.ku.dk)
15-10-2017

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

## Task 3

### a)
	1:	P1 Read A:	cold
	2:	P2 Read B:  cold
	3:	P3 read c:  cold
	4:	P1 write a: hit				// invalidates every other processor with (a,b or c)
	5:	P3 read d:  cold			// evict c since its directly mapped
	6:	P2 read b:  false sharing 	// Block was invalidated by write a by P1, b was not updated though so the correct value was in the cache.
	7:	P1 write b: hit 			// was in shared state because read b from P2 
	8:	P3 read c:  replacement		// since d had evicted c it is replacement miss
	9:	P2 read b:  true sharing	// Block was invalidated by P1 on write b and needs to be brought in from memory. 

### b)
False sharing.
False sharing essentially means that we have the up to date value but it is in a cache block where another element was invalidated

## Task 4

### a)
Read-cache miss memory copy clean
memory block is in shared state
Home = Local

	Local read cache miss:	1 cycle
    Local directory lookup:	50 cycles 

51 cycles
0 trafic

DASH
    nothing to optimize since no hops

### b)

Read-cache miss on dirty memory
Home = local

    Local Read cache miss:						1 cycle
    Local Directory lookup:						50 cycles
    Local Sends Remote read request to Remote: 	20 cycles, 	6 bytes
    Remote lookup cache: 						50 cycles 
    Remote sends reply(flush) to Home: 			100 cycles, 6 bytes + 32bytes
    Home store cache in memory: 				50 cycles

Cycles: 271
Trafic:	44 bytes

DASH:
    nothing to optimize since only 2 hops

### c)
Read-cache miss on clean memory
Home != local

    Local read cache miss: 			1 cycle
    Local remote read to Home: 		20 cycles, 	6 bytes
    Home directory/cache lookup:	50 cycles
    Home flush to Local: 			100 cycles, 6 + 32 bytes
    Local store memory: 			50 cycles

Cycles: 121
Trafic:	44 bytes

DASH:
    nothing to optimize since only 2 hops


### d)
Reach-cache miss on Dirty read
Home != local
Home == remote

    Local read cache miss:			1 cycle
    Local remote read to Home:		20 cycles, 	6 bytes
    Home directory/cache lookup:	50 cycles
    Home flush to Local:			100 cycles, 6 + 32 bytes
    Local store memory:				50 cycles

Cycles:	121 
Trafic:	44 bytes

DASH:
    nothing to optimize since only 2 hops


### e)
Read cache-miss on dirty memory
Home != local != remote

    Local read cache miss:			1 cycle
    Local Remote read to Home:		20 cycles, 	6 bytes
    Home directory/cache lookup:	50 cycles
    Home remote read to Remote: 	20 cycles, 	6 bytes
    Remote directory/cache lookup:	50 cycles
    Remote flush to home: 			100 cycles, 6 + 32 bytes
    Home store:						50 cycles
    Home flush to Local: 			100 cycles, 6 + 32 bytes
    Local store: 					50 cycles

Cycles: 441
Trafic: 88 bytes

DASH:
    4 hops therefore we can optimize.

    Local read cache miss:			1 cycle
    Local Remote read to Home:		20 cycles, 	6 bytes
    Home directory/cache lookup:	50 cycles
    Home remote read to Remote: 	20 cycles, 	6 bytes
    Remote directory/cache lookup:	50 cycles
    Remote flush to home+local:		100 cycles, 2(6 + 32) bytes
    Home store:						50 cycles // can be parallellized with next line
    Local store: 					50 cycles // can be parallelized with previous line

	It should be noted that one could save the trafic from remote to home if home is not required to have the newest value of the memory.
	The way I have written it only saves clock cycles by parallelizing operations. 

## Task 5

### a)
16-by-16 tori 
Diameter:	n		= 16

### b)
Cut					2n				= 32 
Bisect bandwidth 	cut*100 mbits/s = 3200 mbits/s = 3.2 gbits/s

### c)
Total # nodes:  	n^2     		= 256
Total # links:		2*n^2   		= 512
Total bandwidth: 	2n^2*100 		= 51200 = 51.2 gbits/s
Bandwtidth / node 	2n^2*100/n^2 	= n^2*100 = 25600 mbits/s = 25.6 gbits/s

## Task 6

First i find the n-values for the different number of nodes (I denote the number of nodes #N)

	| #N	| 4		| 16	| 64	| 256	|
	| Toru	| 2*2	| 4*4	| 8*8	| 16*16 |
	| Cube	| 2^2	| 2^4	| 2^6	| 2^8	|

### a)
Bisect:
	| Size	| Toru  | Cube	|
	| #N	| 2n	| 2^k-1	|
	| 4		| 4     | 2		|
	| 16	| 8		| 8		|
	| 64	| 16	| 32	|
	| 256 	| 32	| 128	|

Cube provides stricly higher bisection for #N = 64 and higher

### b)
Diameter:
	| Size	| Toru  | Cube	|
	| #N	| n		| k		|
	| 4		| 2     | 2		|
	| 16	| 4		| 4		|
	| 64	| 8		| 6		|
	| 256 	| 16	| 8		|

Switch degree:
	| Size	| Toru  | Cube	|
	| #N	| 4		| k		|
	| 4		| 4     | 2		|
	| 16	| 4		| 4		|
	| 64	| 4		| 6		|
	| 256 	| 4		| 8		|

Therefore the diameter of the cube is lower than the diameter of the n-by-n toru for #N 64 and 256 but the switch degree is higher.

### c)
Having a lower diameter is nice since it describes the maximum number of links required to send information between any two nodes. Furthermore better bisection width is also good because it allows for higher parallelism. Lower switch degree is also nice since that is less resources needed. 
Therefore the cube clearly seem to be the best choice... if not for the physically imposibility of creating it for us 3dimensional creatures.

