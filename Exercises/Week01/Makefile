compile: 
	nvcc -O3 simple.cu;
	futhark-opencl lssp.fut;
	futhark-opencl primes-flat.fut;
	futhark-dataset --i32-bounds=-50:50 -g [1000000]i32 > data;

run:
	./a.out ;
	cat data | ./lssp;
	echo 100 | ./primes-flat;