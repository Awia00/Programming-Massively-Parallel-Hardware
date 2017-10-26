#ifndef SCAN_HOST
#define SCAN_HOST

#include "Kernels.cu.h"
#include "device_launch_parameters.h"

// Week 3: Task 1
template<class T, int tile>
void matrixTranspose(unsigned int block_size,
	float*				d_A,
    float*				d_out,
	const unsigned long M,
	const unsigned long N,
    bool                optimized
) {
	if (optimized) {
		int dimx = ceil(((float)M) / tile);
		int dimy = ceil(((float)N) / tile);
		dim3 block(tile, tile), grid(dimx, dimy);
		matrixTranspose<float, tile> << <grid, block >> > (d_A, d_out, M, N);
	}
	else {
		int dimx = ceil(((float)M) / block_size);
		int dimy = ceil(((float)N) / block_size);
		dim3 block(block_size, block_size), grid(dimx, dimy);
		matrixTransposeNaive<float><< <grid, block >> >(d_A, d_out, M, N);
	}
	cudaThreadSynchronize();
}


// Week 3: Task 2
template<int T>
void squareAccumulator(unsigned int block_size,
    const unsigned long N,
	float*				d_in,
    float*				d_out, 
    bool                optimized
) {
    unsigned int num_blocks = ( (N % block_size) == 0) ?
		N / block_size     :
		N / block_size + 1 ;

    int size = N*T;
    float *d_in_trans;
    float *d_out_trans;
	cudaMalloc((void**)&d_in_trans, size * sizeof(float));
	cudaMalloc((void**)&d_out_trans, size * sizeof(float));

    if(optimized) {
		matrixTranspose<float, T>(block_size, d_in, d_in_trans, N, T, true);
		squareAccumulatorTranspose<<<num_blocks, block_size>>>(d_in_trans, d_out_trans, N);
        matrixTranspose<float, T>(block_size, d_out_trans, d_out, N, T, true);
    } else {
		squareAccumulator<<<num_blocks, block_size >>>(d_in, d_out, N);
    }
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
		dim3 block(T, T, 1), grid(dimx, dimy, 1);
		
		matrixMatrixMul<T> << <grid, block >> >(d_A, d_B, d_C, M, N, U);
	}
	else {
		int dimy = ceil(((float)M) / block_size);
		int dimx = ceil(((float)N) / block_size);
		dim3 block(block_size, block_size, 1), grid(dimx, dimy, 1);
		
		matrixMatrixMulNaive<< <grid, block >> >(d_A, d_B, d_C, M, N, U, block_size);
	}
	cudaThreadSynchronize();
}
#endif //SCAN_HOST


