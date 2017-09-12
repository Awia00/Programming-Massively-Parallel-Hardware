#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void square(float * d_out, float * d_in) {
    const unsigned int lid = threadIdx.x;
    const unsigned int gid = blockIdx.x*blockDim.x + lid;
    float f = d_in[gid];
    d_out[gid] = f * f;
}

__global__ void cube(float * d_out, float * d_in) {
    const unsigned int lid = threadIdx.x;
    const unsigned int gid = blockIdx.x*blockDim.x + lid;
    float f = d_in[gid];
    d_out[gid] = f * f * f;
}

__global__ void special(float * d_out, float * d_in) {
    const unsigned int lid = threadIdx.x;
    const unsigned int gid = blockIdx.x*blockDim.x + lid;
    float f = d_in[gid];
    d_out[gid] = f * f * f;
}

int main(int argc, char **argv) {
    const int num_threads = 1024;
	const int ARRAY_SIZE = 8192;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
	
	// Generate the input array on the host
	float * h_in = (float*) malloc(ARRAY_BYTES);
    float * h_out = (float*) malloc(ARRAY_BYTES);
    
    for(int i = 0; i<ARRAY_SIZE; i++){
        h_in[i] = float(i);
    }

    // declare GPU memory pointers
    float * d_in;
    float * d_out;

    // allocate GPU memory
    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_out, ARRAY_BYTES);

    // copy CPU memory to GPU memory
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    // Launch the kernel
    square<<<ARRAY_SIZE/num_threads, num_threads>>>(d_out, d_in);

    // copy back the result
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    
    // print result
    for(int i = 0; i<ARRAY_SIZE; i++){
        printf("%f", h_out[i]);
        printf((i % 4 != 3) ? "\t" : "\n");
    }

    // free cpu memory
    free(h_in);
    free(h_out);
    // free up GPU memory
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}