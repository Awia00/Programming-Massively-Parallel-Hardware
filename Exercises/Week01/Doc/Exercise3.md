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

