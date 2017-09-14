compile: 
	nvcc -O3 simple.cu \
	futhark-opencl lssp.fut \
	futhark-opencl primes-flat.fut

run:
	./a.out \
	data | ./lssp 
	./primes-flat