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

