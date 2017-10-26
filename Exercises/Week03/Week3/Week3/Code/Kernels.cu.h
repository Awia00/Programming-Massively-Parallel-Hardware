#ifndef SCAN_KERS
#define SCAN_KERS

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

// Week 3 task 1
template <class T>
__global__ void matrixTransposeNaive(float* A, float* transposedA, int M, int N ) {
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int j = blockIdx.x*blockDim.x + tidx;
    int i = blockIdx.y*blockDim.y + tidy;

    if( j < N && i < M )
        transposedA[j*M + i] = A[i*N+j];
}

// Week 3 task 1
template <class T, int TILE>
__global__ void matrixTranspose(float* A, float* trA, int M, int N ) {
    __shared__ float tile[TILE][TILE +1];
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int j = blockIdx.x*TILE + tidx;
    int i = blockIdx.y*TILE + tidy;

	if (j < N && i < M)
        tile[tidy][tidx] = A[i*N+j];
    
    __syncthreads();
    
    i = blockIdx.y*TILE + tidy;
    j = blockIdx.x*TILE + tidx;
    if( j < N && i < M )
        trA[j*M+i] = tile[tidx][tidy];
}

// Week 3 task 2
__global__ void squareAccumulator(float* A, float* B, int N) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
	if (gid < N) {
		int i = gid * 64;
		float accum = A[i] * A[i];
		B[gid * 64] = accum;
		for (int j = 1; j < 64; j++) {
			float tmpA = A[i + j];
			accum = sqrt(B[i + j - 1]) + tmpA*tmpA;
			B[i + j] = accum;
		}
	}
}

// Week 3 task 2
__global__ void squareAccumulatorTranspose(float* Atrans, float* B, int N) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
	if (gid < N) {
		int i = gid * 64;
		float accum = Atrans[gid] * Atrans[gid];
		B[i] = accum;
		for (int j = 1; j < 64; j++) {
			float tmpA = Atrans[j*N + gid];
			accum = sqrt(B[i + j - 1]) + tmpA*tmpA;
			B[i + j] = accum;
		}
	}
}

// Week 3 task 3
__global__ void matrixMatrixMulNaive(float* A, float* B, float* C, int N, int M, int U, int T) {
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int j = blockIdx.x*T + tidx;
    int i = blockIdx.y*T + tidy;

    float tmp = 0.0f;
    if( j < N && i < M ) {
        for(int k = 0; k < U; k++) {
			tmp += A[i*U + k] * B[k*N + j];
        }
    }
    C[i*N + j] = tmp;
}

// Week 3 task 3
template <int T> // KERNEL
__global__ void matrixMatrixMul(float*A, float* B, float* C, int M, int N, int U ) {
    __shared__ float Ash[T][T], Bsh[T][T];
    int ii = blockIdx.y * T; //blockDim.x==T
    int jj = blockIdx.x * T; //blockDim.y==T
    int tidy = threadIdx.y, i = tidy+ii;
    int tidx = threadIdx.x, j = tidx+jj;
    float tmp = 0.0;
    for(int kk=0; kk<U; kk+=T) {
        Ash[tidy][tidx] = (i<M && kk+tidx<U) ? A[i*U + (kk+tidx)] : 0.0;
        Bsh[tidy][tidx] = (j<N && kk+tidy<U) ? B[(kk+tidy)*N + j] : 0.0;
		__syncthreads();
        
        for(int k=0; k<T; k++) {
			tmp += Ash[tidy][k] * Bsh[k][tidx];
        } 
		__syncthreads();
    } 
    if (i<M && j<N) C[i*N + j] = tmp;
}


#endif //SCAN_KERS

