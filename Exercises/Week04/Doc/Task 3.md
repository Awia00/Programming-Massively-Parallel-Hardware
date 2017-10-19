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

