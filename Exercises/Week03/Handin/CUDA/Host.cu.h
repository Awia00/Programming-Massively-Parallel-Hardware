#ifndef SCAN_HOST
#define SCAN_HOST

#include "Kernels.cu.h"
#include "device_launch_parameters.h"

// Week 3: Task 1
template<int tile>
void matrixTranspose(unsigned int block_size,
	float*				d_A,
    float*				d_out,
	const unsigned long M,
	const unsigned long N,
    bool                optimized
) {
	if (optimized) {
		int dimy = ceil(((float)M) / tile);
		int dimx = ceil(((float)N) / tile);
		dim3 block(tile, tile), grid(dimx, dimy);
		matrixTranspose<tile> << <grid, block >> > (d_A, d_out, M, N);
	}
	else {
		int dimy = ceil(((float)M) / block_size);
		int dimx = ceil(((float)N) / block_size);
		dim3 block(block_size, block_size), grid(dimx, dimy);
		matrixTransposeNaive << <grid, block>>> (d_A, d_out, M, N);
	}
	int error = cudaGetLastError();
	if (cudaSuccess != error)
		printf("Failed in matrixTransposeKernel with error code: %d\n", error);

	cudaThreadSynchronize();
}


// Week 3: Task 2
template<int T>
void squareAccumulator(unsigned int block_size,
    const unsigned long N,
	const unsigned long M,
	float*				d_in,
    float*				d_out, 
    bool                optimized
) {
	unsigned int num_blocks = ceil(N / block_size);

    int size = N*M;
    float *d_in_trans;
    float *d_out_trans;
	cudaMalloc((void**)&d_in_trans, size * sizeof(float));
	cudaMalloc((void**)&d_out_trans, size * sizeof(float));

    if(optimized) {
		matrixTranspose<32>(32, d_in, d_in_trans, N, M, true);
		squareAccumulatorTranspose<<<num_blocks, block_size>>>(d_in_trans, d_out_trans, N, M);
        matrixTranspose<32>(32, d_out_trans, d_out, M, N, true);
    } else {
		squareAccumulator<<<num_blocks, block_size >>>(d_in, d_out, N, M);
    }
	int error = cudaGetLastError();
	if (cudaSuccess != error)
		printf("Failed in squareAccumulatorKernel with error code: %d\n", error);

	cudaThreadSynchronize();
}

// Week 3: Task 3
template<int T>
void matrixMatrixMul(
	float*				d_A,
	float*				d_B,
    float*				d_C,
	const unsigned long M,
	const unsigned long N,
	const unsigned long U,
	unsigned int block_size,
    bool optimized
) {
	if (optimized) {
		int dimy = ceil(((float)M) / T);
		int dimx = ceil(((float)N) / T);
		dim3 block(T, T), grid(dimx, dimy);
		
		matrixMatrixMul<T> << <grid, block >> >(d_A, d_B, d_C, M, N, U);
	}
	else {
		int dimy = ceil(((float)M) / block_size);
		int dimx = ceil(((float)N) / block_size);
		dim3 block(block_size, block_size), grid(dimx, dimy);
		
		matrixMatrixMulNaive<< <grid, block >> >(d_A, d_B, d_C, M, N, U, block_size);
	}
	int error = cudaGetLastError();
	if (cudaSuccess != error)
		printf("Failed in matrixMatrixMulKernel with error code: %d\n", error);

	cudaThreadSynchronize();
}
#endif //SCAN_HOST


