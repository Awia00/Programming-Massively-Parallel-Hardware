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

