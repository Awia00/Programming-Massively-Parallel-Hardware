# Exercises Week 1
*Anders Wind Steffensen* (qrj742@alumni.ku.dk)
18-09-2017

## Exercise 1

Theorems:
Theorem 1: 	(map f) . (map g) = map(f . g)
Theorem 2: 	(map f) . (reduce (++) []) = (redude(++) []) . (map(map f))
Theorem 3: 	(reduce op id) . (reduce (++) []) = (reduce op id) . (map(reduce op id))

distr_p :: [a]->[[a]]

Redomap: redomap op f id = (reduce op id) . (map f)
Hint: (reduce (++) []) . distrp = id


Proof: 
We want to prove 
	redomap op f id = (reduce op id) . (map(redomap op f id)) . distrp

redomap op f id = (reduce op id) . (map f)
				= (reduce op id) . (map f) . id
				= (reduce op id) . (map f) . (reduce (++) []) . distrp  		: by hint
				= (reduce op id) . (reduce (++) []) . (map(map f)) . distrp  	: by theorem 2
				= (reduce op id) . (map(reduce op id)) . (map(map f)) . distrp  : by theorem 3
				= (reduce op id) . (map((reduce op id) . (map f)) . distrp  	: by theorem 1
Proof done.

## Exercise 2

For implementation see file "lssp.fut"


To meassure the performance of lssp, I generated the following dataset:
    $ futhark-dataset --i32-bounds=-50:50 -g [1000000]i32 > data

I then ran(multiple times) both the futhark-c compiled and the futhark-opencl compiled versions and get the following results:
    GPU (opencl)
    $ ./lssp -t /dev/stderr
    1720
    10i32

    CPU (c)
    2082
    10i32

Where the 10i32 is the result of lssp. 

We can see here that the running time of the opencl compiled version is faster than the cpu version. The speedup is not just a factor of adding the cores of the machine, which means that it is probably limited by the accesses to global memory. Had we increased the size of the array we would probably see a larger speedup in the GPU version.

## Exercise 3

Program:
    sqrt_primes = primesOpt (sqrt (fromIntegral n))
    nested = map (\p -> 
        let m = (n 'div' p)
        in map (\j -> j*p) [2..m])
    not_primes = reduce (++) [] nested

Normalized:
    sqrt_primes = primesOpt (sqrt (fromIntegral n))
    nested = map (\p ->
        let m       = n ‘div‘ p in              -- distribute map   // 1
        let mm1     = m - 1 in                  -- distribute map   // 2
        let iot     = iota mm1 in               -- F rule 4         // 3
        let twom    = map (+2) iot in           -- F rule 2         // 4
        let rp      = replicate mm1 p in        -- F rule 5         // 5
        in map (\(j,p) -> j*p) (zip twom rp)    -- F rule 2         // 6
    ) sqrt_primes


Flattened.
    sqrt_primes = primesOpt (sqrt (fromIntegral n))
    F( map (\p -> let m = (n 'div' p) in map (\j -> j*p) [2..m]) ) ==
        1. let ms       =  map(\p -> n ‘div‘ p) sqrt_primes
        2. let mm1s     = map(\m -> m - 1) ms
        3. let iots     = F( map(\mm1 -> (iota mm1) mm1s) )
        4. let twoms    = F( map(\iot -> map (+2) iot) iots )
        5. let rps      = F( map (\(mm1, p) -> replicate mm1 p) mm1s sqrt_primes )
        6. let nested   = F(map(\(js,ps) -> map (*) js ps)) twoms rps -- assuming map goes over each element of the twoms and rps at the same time.

    3: using rule 4
    F( map(\mm1 -> (iota mm1) mm1s )
        inds = scanexc (+) 0 mmis
        size = reduce (+) 0 mmis
        flag = scatter (replicate size 0) inds arr
        tmp = replicate size 1
        iots = sgmScanExc (+) 0 flag tmp

    4: using rule 2
    F( map(\iot -> map (+2) iot) iots )
        twoms = map(\i -> i +2) iots

    5: using rule 5
    F( map (\(mm1, p) -> replicate mm1 p) mm1s sqrt_primes )
        vals = scatter (replicate size 0) inds sqrt_primes
        rps = sgmScanInc (+) flag vals

    6: using rule 2
    F(map(\(js,ps) -> map (*) js ps)) twoms rps
        nested = map (*) twoms rps


Final version:
    
    sqrt_primes = primesOpt (sqrt (fromIntegral n))
    ms      = map(\p -> n ‘div‘ p) sqrt_primes
    mm1s    = map(\m -> m - 1) ms
    inds    = scanexc (+) 0 mmis
    size    = reduce (+) 0 mmis
    flag    = scatter (replicate size 0) inds arr
    tmp     = replicate size 1
    iots    = sgmScanExc (+) 0 flag tmp
    twoms   = map(\i -> i +2) iots
    vals    = scatter (replicate size 0) inds sqrt_primes
    rps     = sgmScanInc (+) flag vals
    not_primes  = map (*) twoms rps -- nested is not_primes since it is already flattened
    mm      = length not_primes


For the implementation, see "primes-flat.fut"

## Exercise 4

For implementation, see file "simple.cu"

I typically see such results:
    
    CPU Took 132 microseconds (0.13ms)
    GPU Took 46 microseconds (0.05ms)

The GPU is a factor of ~3 faster than the CPU. Even though the CPU is running sequentially, the faster clock speed and cache control makes it relatively fast at computing the result. The GPU however while faster is not faster by a factor of the number of cores. The clock speed of each core is of course much lower but the explanation to the relatively low speedup is probably that the GPU spends too much time on retrieving/writing to the global memory that is the math/memory ratio is too low.