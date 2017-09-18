## Exercise 3

### Program

    sqrt_primes = primesOpt (sqrt (fromIntegral n))
    nested = map (\p -> 
        let m = (n 'div' p)
        in map (\j -> j*p) [2..m])
    not_primes = reduce (++) [] nested

### Normalized

    sqrt_primes = primesOpt (sqrt (fromIntegral n))
    nested = map (\p ->
        let m       = n ‘div‘ p in              -- distribute map   // 1
        let mm1     = m - 1 in                  -- distribute map   // 2
        let iot     = iota mm1 in               -- F rule 4         // 3
        let twom    = map (+2) iot in           -- F rule 2         // 4
        let rp      = replicate mm1 p in        -- F rule 5         // 5
        in map (\(j,p) -> j*p) (zip twom rp)    -- F rule 2         // 6
    ) sqrt_primes


### Flattened

    sqrt_primes = primesOpt (sqrt (fromIntegral n))
    F( map (\p -> let m = (n 'div' p) in map (\j -> j*p) [2..m]) ) =
        1. let ms       = map(\p -> n ‘div‘ p) sqrt_primes
        2. let mm1s     = map(\m -> m - 1) ms
        3. let iots     = F( map(\mm1 -> (iota mm1) mm1s) )
        4. let twoms    = F( map(\iot -> map (+2) iot) iots )
        5. let rps      = F( map (\(mm1, p) -> replicate mm1 p) mm1s sqrt_primes )
        6. let nested   = F(map(\(js,ps) -> map (*) js ps)) twoms rps -- assuming automatic zipping of elements in twoms and rps.

    3: using rule 4
    F( map(\mm1 -> (iota mm1) mm1s )
        inds = scanexc (+) 0 mmis
        size = reduce (+) 0 mmis
        flag = scatter (replicate size 0) inds mm1s
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


For the implementation, see "primes-flat.fut"


### Discussion

    primes-naive 	(CPU): 
        $ echo 10000000 | ./primes-naive -t /dev/stderr > /dev/null
        593387
    primes-naive 	(GPU):
        echo 10000000 | ./primes-naive -t /dev/stderr > /dev/null
        38996
    primes-opt 		(CPU):
        $ echo 10000000 | ./primes-opt -t /dev/stderr > /dev/null
        229836
    primes-opt 		(GPU):
         $ echo 10000000 | ./primes-opt -t /dev/stderr > /dev/null
         **Does not finish within 1 min**
    primes-flat		(CPU):
        $ echo 10000000 | ./primes-flat -t /dev/stderr > /dev/null
        329925
    primes-flat		(GPU):
        echo 10000000 | ./primes-flat -t /dev/stderr > /dev/null
        98041

I am generally seeing some weird results. First of all to compare the CPU versions; *Naive* is the slowest version which makes sense since the depth is "sqrt(n)" and not "lg lg n" like *opt* and *flat*. Naturally we also see an increase in runnning time for *flat* since the flattening requires more operations. The GPU results are difficult to explain. *Naive* is the fastest of the three with an order of magnitude better running time than its own CPU version. The *flat* GPU version is also an improvement over the CPU version but not to the same extend. In *opt* an "unsafe"-keyword had to be added around the last filter to make it compile with opencl. The program did not finish within a reasonable time limit so I omitted the results. I would have expected *flat* to be the fastest of the three given the lower depth and since the flattened form would allow for higher parallelism. 

