## Exercise 1

Theorems:
Theorem 1: 	(map f) . (map g) = map(f . g)
Theorem 2: 	(map f) . (reduce (++) []) = (redude(++) []) . (map(map f))
Theorem 3: 	(reduce op id) . (reduce (++) []) = (reduce op id) . (map(reduce op id))

distr_p :: [a]->[[a]]

Redomap: redomap op f id = (reduce op id) . (map f)
Hint: (reduce (++) []) . distrp = id

Proof: 
We want to prove:

	redomap op f id = (reduce op id) . (map(redomap op f id)) . distrp


	redomap op f id = (reduce op id) . (map f)
			= (reduce op id) . (map f) . id
			= (reduce op id) . (map f) . (reduce (++) []) . distrp  	: by hint
			= (reduce op id) . (reduce (++) []) . (map(map f)) . distrp  	: by theorem 2
			= (reduce op id) . (map(reduce op id)) . (map(map f)) . distrp  : by theorem 3
			= (reduce op id) . (map((reduce op id) . (map f)) . distrp  	: by theorem 1
Proof done.

