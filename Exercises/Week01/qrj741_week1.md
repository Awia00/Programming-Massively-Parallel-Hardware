# Exercises Week 1
*Anders Wind Steffensen* (qrj742@alumni.ku.dk)

## Exercise 1

	Theorems:
	Theorem 1: 	(map f) . (map g) = map(f . g)
	Theorem 2: 	(map f) . (reduce (++) []) = (redude(++) []) . (map(map f))
	Theorem 3: 	(reduce op id) . (reduce (++) []) = (reduce op id) . (map(reduce op id))
	
	Distrp
	distr_p :: [a]->[[a]]

	Redomap
	Redomap: 	redomap op f id = (reduce op id) . (map f)
	Hint: 		(reduce (++) []) . distrp = id

	Rule 2: 
	Rule 3: redomap op f id = (reduce op id) . (map(redomap op f id)) . distrp


	Proof: 
	We want to prove that "redomap op f id = (reduce op id) . (map(redomap op f id)) . distrp"

	redomap op f id = (reduce op id) . (map f)
					= (reduce op id) . (map f) . id
					= (reduce op id) . (map f) . (reduce (++) []) . distrp  		: by hint
					= (reduce op id) . (reduce (++) []) . (map(map f)) . distrp  	: by theorem 2
					= (reduce op id) . (map(reduce op id)) . (map(map f)) . distrp  : by theorem 3
					= (reduce op id) . (map((reduce op id) . (map f)) . distrp  	: by theorem 1
	Proof done.

## Exercise 2

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
    not_primes = reduce (++) [] nested

    3: using rule 4
    F( map(\mm1 -> (iota mm1) mm1s )
        inds = scanexc (+) 0 mmis
        size = reduce (+) 0 mmis
        flag = scatter (replicate size 0) inds arr
        tmp = replicate size 1
        iots = sgmScanExc (+) 0 flag tmp

    4: using rule 2
    F( map(\iot -> map (+2) iot) iots )
        map(\i -> i +2) iots

    5: using rule 5
    F( map (\(mm1, p) -> replicate mm1 p) mm1s sqrt_primes )
        vals = scatter (replicate size 0) inds sqrt_primes
        rps = sgmScanInc (+) flag vals

    6: using rule 2
    F(map(\(js,ps) -> map (*) js ps)) twoms rps
        map (*) twoms rps


## Exercise 4

