#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "Host.cu.h"

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
#include "TimeOfDay.h" // Only on windows
#else
#include <sys/time.h> // Only on linux
#endif


int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
	unsigned int resolution = 1000000;
	long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
	result->tv_sec = diff / resolution;
	result->tv_usec = diff % resolution;
	return (diff<0);
}


void matrixTranspose(float* matrix, float* outMatrix, int M, int N) {
	for(int i = 0; i < M; i++) {
		for(int j = 0; j < N; j++){
			outMatrix[j*M + i] = matrix[i*N + j];
		}
	}
}

void matrixTransposeTest(bool optimized) {
	if(optimized)
		printf("\nRunning Matrix Transpose optimized\n");
	else
		printf("\nRunning Matrix Transpose naive\n");

	const unsigned int block_size = 512;
	const unsigned int T = 64;
	const unsigned int N = 8000;
	const unsigned int M = 8000;
	unsigned int mem_size_A = M*N * sizeof(float);	
	float* h_A = (float*)malloc(mem_size_A);
	float* h_A_out = (float*)malloc(mem_size_A);

	for(int i = 0; i<M; i++) {
		for(int j = 0; j<N; j++){
			h_A[i*N + j] = rand() % 100;
		}
	}

	float* d_A;
	float* d_A_out;
	cudaMalloc((void**)&d_A, mem_size_A);
	cudaMalloc((void**)&d_A_out, mem_size_A);
	// copy host memory to device
	cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_A_out, h_A_out, mem_size_A, cudaMemcpyHostToDevice);

	unsigned long int elapsed;
	struct timeval t_start,t_end,t_diff;
	gettimeofday(&t_start, NULL);
	
	// run 
	matrix_transpose<T>(block_size, M, N, d_A, d_A_out, optimized);
	cudaThreadSynchronize();
	
	gettimeofday(&t_end, NULL);
	timeval_subtract(&t_diff, &t_end, &t_start);
	elapsed = (t_diff.tv_sec*1e6 + t_diff.tv_usec);
	
	cudaMemcpy(h_A_out, d_A_out, mem_size_A, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_A_out);

	// print results
	if (optimized) {
		printf("M-transpose optimized time\t %d\n", elapsed);
	}
	else {
		printf("M-transpose naive time\t\t %d\n", elapsed);
	}

	free(h_A);
	free(h_A_out);
}

void squareAccumulatorTest(bool optimized) {
	if (optimized)
		printf("\nRunning Square Accumulator optimized\n");
	else
		printf("\nRunning Square Accumulator naive\n");

	const unsigned int block_size = 512;
	const unsigned int T = 64;
	const unsigned int N = 250000;
	unsigned int mem_size = N*T * sizeof(float);	
	float* h_A = (float*)malloc(mem_size);
	float* h_B = (float*)malloc(mem_size);

	for(int i = 0; i<N; i++) {
		for(int j = 0; j<T; j++){
			h_A[i*T + j] = rand() % 100;
		}
	}

	float* d_A;
	float* d_B;
	cudaMalloc((void**)&d_A, mem_size);
	cudaMalloc((void**)&d_B, mem_size);
	// copy host memory to device
	cudaMemcpy(d_A, h_A, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, mem_size, cudaMemcpyHostToDevice);

	unsigned long int elapsed;
	struct timeval t_start,t_end,t_diff;
	gettimeofday(&t_start, NULL);
	
	// run 
	square_accumulator<T>(block_size, N, d_A, d_B, optimized);
	cudaThreadSynchronize();
	
	gettimeofday(&t_end, NULL);
	timeval_subtract(&t_diff, &t_end, &t_start);
	elapsed = (t_diff.tv_sec*1e6 + t_diff.tv_usec);

	cudaMemcpy(h_B, d_B, mem_size, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);

	// print results
	if (optimized)
		printf("Square Accumulator optimized time\t %d\n", elapsed);
	else
		printf("Square Accumulator naive time\t\t %d\n", elapsed);

	free(h_A);
	free(h_B);
}

void matrixMatrixMul(float* A, float* B, float* C, int N, int M, int U) {
	for(int i = 0; i < M; i++) {
		for(int j = 0; j < N; j++) {
			float tmp = 0.0f;
			for(int k = 0; k < U; k++)
				tmp += A[i*M + k] * B[k*U + j];
			C[i*M + j] = tmp;
		}
	}
}

//  A, B and C have sizes MxU, UxN and MxN
void matrixMatrixMulTest(bool optimized){
	if (optimized)
		printf("\nRunning Matrix Matrix Mul optimized\n");
	else
		printf("\nRunning Matrix Matrix Mul naive\n");

	const unsigned int block_size = 512;
	const unsigned int T = 64;
	const unsigned int N = 8000;
	const unsigned int M = 8000;
	const unsigned int U = 8000;
	unsigned int mem_size_A = M*U * sizeof(float);	
	unsigned int mem_size_B = U*N * sizeof(float);	
	unsigned int mem_size_C = M*N * sizeof(float);
	float* h_A = (float*)malloc(mem_size_A);
	float* h_B = (float*)malloc(mem_size_B);
	float* h_C = (float*)malloc(mem_size_C);

	for(int i = 0; i<M; i++) {
		for(int j = 0; j<U; j++){
			h_A[i*M + j] = rand() % 100;
		}
	}
	for(int i = 0; i<U; i++) {
		for(int j = 0; j<N; j++){
			h_B[i*U + j] = rand() % 100;
		}
	}
	for(int i = 0; i<M; i++) {
		for(int j = 0; j<N; j++){
			h_C[i*M + j] = rand() % 100;
		}
	}

	float* d_A;
	float* d_B;
	float* d_C;
	cudaMalloc((void**)&d_A, mem_size_A);
	cudaMalloc((void**)&d_B, mem_size_B);
	cudaMalloc((void**)&d_C, mem_size_C);
	// copy host memory to device
	cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, mem_size_C, cudaMemcpyHostToDevice);


	unsigned long int elapsed;
	struct timeval t_start,t_end,t_diff;
	gettimeofday(&t_start, NULL);
	
	// run 
	matrix_matrix_mul<T>(block_size, M, N, U, d_A, d_B, d_C, optimized);
	cudaThreadSynchronize();
	
	gettimeofday(&t_end, NULL);
	timeval_subtract(&t_diff, &t_end,&t_start);
	elapsed = (t_diff.tv_sec*1e6 + t_diff.tv_usec);
	double flops = 2.0 * M * N * U;
	double gigaFlops=(flops*1.0e-3f) /elapsed;
	
	cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	// Print the results
	if (optimized) {
		printf("MMM optimized time\t\t %d\n", elapsed);
		printf("MMM gigaFlops optimized\t %f\n", gigaFlops);
	}
	else {
		printf("MMM naive time\t\t %d\n", elapsed);
		printf("MMM gigaFlops naive\t %f\n", gigaFlops);
	}

	free(h_A);
	free(h_B);
	free(h_C);
}

int main(int argc, char** argv) {
	printf("\n");

	matrixTransposeTest(false);
	printf("\n==========================\n");
	matrixTransposeTest(true);

	printf("\n=========================="); 
	printf("\n==========================\n");

	squareAccumulatorTest(false);
	printf("\n==========================\n");
	squareAccumulatorTest(true);

	printf("\n=========================="); 
	printf("\n==========================\n");

	matrixMatrixMulTest(false);
	printf("\n==========================\n");
	matrixMatrixMulTest(true);
}

