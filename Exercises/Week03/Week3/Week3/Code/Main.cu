#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#if defined(_OPENMP)
#include "omp.h"
#else
#include "Host.cu.h"
#endif

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
#include "TimeOfDay.h" // Only on windows
#else
#include <sys/time.h> // Only on linux
#include "Main.h"
#endif




int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
	unsigned int resolution = 1000000;
	long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
	result->tv_sec = diff / resolution;
	result->tv_usec = diff % resolution;
	return (diff<0);
}

#pragma region MatrixTranspose  

void matrixTransposeSequential(float* matrix, float* outMatrix, int M, int N) {
	for(int i = 0; i < M; i++) {
		for(int j = 0; j < N; j++){
			outMatrix[j*M + i] = matrix[i*N + j];
		}
	}
}

#if defined(_OPENMP)

void matrixTransposeOMP(float* matrix, float* outMatrix, int M, int N) {
#pragma omp parallel shared(matrix, outMatrix, M, N) default(none)
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			outMatrix[j*M + i] = matrix[i*N + j];
		}
	}
}


void matrixTransposeOMPTest() {
	printf("\nRunning Matrix Transpose OMP\n");

	const unsigned int N = 8000;
	const unsigned int M = 8000;
	unsigned int mem_size_A = M*N * sizeof(float);
	float* h_A = (float*)malloc(mem_size_A);
	float* h_A_out = (float*)malloc(mem_size_A);
	float* h_A_expected = (float*)malloc(mem_size_A);

	for (int i = 0; i<M; i++) {
		for (int j = 0; j<N; j++) {
			h_A[i*N + j] = rand() % 100;
		}
	}

	matrixTransposeOMP(h_A, h_A_out, M, N);
	matrixTransposeSequential(h_A, h_A_expected, M, N);

	bool succeded;
	for (int i = 0; i<M; i++) {
		for (int j = 0; j<N; j++) {
			if (h_A_out[i*N + j] != h_A_expected[i*N + j]) {
				printf("\nFailed. Expected %f, was %f\n", h_A_expected[i*N + j], h_A_out[i*N + j]);
				succeded = false;
				break;
			}
		}
		if (!succeded)
			break;
	}
	if (succeded)
		printf("\nCompleted successfully\n");

	free(h_A);
	free(h_A_out);
}

#else

void matrixTransposeGPU(unsigned int mem_size_A, float * h_A, float * h_out, const unsigned int &M, const unsigned int &N, bool optimized)
{
	const unsigned int block_size = 512;
	const unsigned int T = 64;

	float* d_A;
	float* d_out;
	cudaMalloc((void**)&d_A, mem_size_A);
	cudaMalloc((void**)&d_out, mem_size_A);
	// copy host memory to device
	cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);

	unsigned long int elapsed;
	struct timeval t_start, t_end, t_diff;
	gettimeofday(&t_start, NULL);

	// run 
	matrixTranspose<float, T>(block_size, d_A, d_out, M, N, optimized);
	
	cudaMemcpy(h_out, d_out, mem_size_A, cudaMemcpyDeviceToHost);

	gettimeofday(&t_end, NULL);
	timeval_subtract(&t_diff, &t_end, &t_start);
	elapsed = (t_diff.tv_sec*1e6 + t_diff.tv_usec);

	cudaFree(d_A);
	cudaFree(d_out);

	// print results
	if (optimized) {
		printf("M-transpose optimized time\t %d\n", elapsed);
	}
	else {
		printf("M-transpose naive time\t\t %d\n", elapsed);
	}
}

void matrixTransposeGPUTest(bool optimized) {
	if(optimized)
		printf("\nRunning Matrix Transpose optimized\n");
	else
		printf("\nRunning Matrix Transpose naive\n");

	const unsigned int N = 5000;
	const unsigned int M = 4000;
	unsigned int mem_size_A = M*N * sizeof(float);	
	float* h_A = (float*)malloc(mem_size_A);
	float* h_A_out = (float*)malloc(mem_size_A);
	float* h_A_expected = (float*)malloc(mem_size_A);

	for(int i = 0; i<M; i++) {
		for(int j = 0; j<N; j++){
			h_A[i*N + j] = rand() % 100;
		}
	}

	matrixTransposeGPU(mem_size_A, h_A, h_A_out, M, N, optimized);
	matrixTransposeSequential(h_A, h_A_expected, M, N);

	bool succeded;
	for (int i = 0; i<M; i++) {
		for (int j = 0; j<N; j++) {
			if (h_A_out[i*N + j] != h_A_expected[i*N + j]) {
				printf("\nFailed. Expected %f, was %f\n", h_A_expected[i*N + j], h_A_out[i*N + j]);
				succeded = false;
				break;
			}
		}
		if (!succeded)
			break;
	}
	if (succeded)
		printf("\nCompleted successfully\n");

	free(h_A);
	free(h_A_out);
}



#endif

#pragma endregion MatrixTranspose

#pragma region squareAccumulator

void squareAccumulatorSequential(float* A, float* B, int N) {
	int M = 64;
	for (int i = 0; i < N; i++) {
		float accum = A[i*M];
		B[i*M] = accum;
		for (int j = 1; j < 64; j++) {
			float tmpA = A[i*M + j];
			accum = sqrt(accum) + tmpA*tmpA;
			B[i*M + j] = accum;
		}
	}
}

#if defined(_OPENMP)

void squareAccumulatorOMP(float* A, float* B, int N) {
	const int M = 64;
	#pragma omp parallel shared(A,B,N) default(none)
	for (int i = 0; i < N; i++) {
		float accum = A[i*M];
		B[i*M] = accum;
		for (int j = 1; j < 64; j++) {
			float tmpA = A[i*M + j];
			accum = sqrt(accum) + tmpA*tmpA;
			B[i*M + j] = accum;
		}
	}
}

void squareAccumulatorOMPTest() {
	printf("\nRunning Square Accumulator optimized\n");

	const unsigned int T = 64;
	const unsigned int N = 200000;
	unsigned int mem_size = N*T * sizeof(float);
	float* h_A = (float*)malloc(mem_size);
	float* h_B = (float*)malloc(mem_size);
	float* h_B_expected = (float*)malloc(mem_size);

	for (int i = 0; i<N; i++) {
		for (int j = 0; j<T; j++) {
			h_A[i*T + j] = rand() % 100;
		}
	}

	squareAccumulatorOMP(h_A, h_B, N);
	squareAccumulatorSequential(h_A, h_B_expected, N);

	bool succeded;
	for (int i = 0; i<N; i++) {
		for (int j = 0; j<T; j++) {
			if (h_B[i*N + j] != h_B_expected[i*N + j]) {
				printf("\nFailed. Expected %f, was %f\n", h_B_expected[i*N + j], h_B[i*N + j]);
				succeded = false;
				break;
			}
		}
		if (!succeded)
			break;
	}
	if (succeded)
		printf("\nCompleted successfully\n");

	free(h_A);
	free(h_B);
}

#else

void squareAccumulatorGPU(unsigned int mem_size, float * h_A, float * h_B, const unsigned int &N, bool optimized)
{
	float* d_A;
	float* d_B;
	const unsigned int block_size = 512;
	const unsigned int T = 64;
	cudaMalloc((void**)&d_A, mem_size);
	cudaMalloc((void**)&d_B, mem_size);
	// copy host memory to device
	cudaMemcpy(d_A, h_A, mem_size, cudaMemcpyHostToDevice);

	unsigned long int elapsed;
	struct timeval t_start, t_end, t_diff;
	gettimeofday(&t_start, NULL);

	// run 
	squareAccumulator<T>(block_size, N, d_A, d_B, optimized);
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
}

void squareAccumulatorGPUTest(bool optimized) {
	if (optimized)
		printf("\nRunning Square Accumulator optimized\n");
	else
		printf("\nRunning Square Accumulator naive\n");

	const unsigned int T = 64;
	const unsigned int N = 200000;
	unsigned int mem_size = N*T * sizeof(float);	
	float* h_A = (float*)malloc(mem_size);
	float* h_B = (float*)malloc(mem_size);
	float* h_B_expected = (float*)malloc(mem_size);

	for(int i = 0; i<N; i++) {
		for(int j = 0; j<T; j++){
			h_A[i*T + j] = rand() % 100;
		}
	}

	squareAccumulatorGPU(mem_size, h_A, h_B, N, optimized);
	squareAccumulatorSequential(h_A, h_B_expected, N);

	bool succeded;
	for (int i = 0; i<N; i++) {
		for (int j = 0; j<T; j++) {
			if (h_B[i*N + j] != h_B_expected[i*N + j]) {
				printf("\nFailed. Expected %f, was %f\n", h_B_expected[i*N + j], h_B[i*N + j]);
				succeded = false;
				break;
			}
		}
		if (!succeded)
			break;
	}
	if (succeded)
		printf("\nCompleted successfully\n");

	free(h_A);
	free(h_B);
}

#endif

#pragma endregion SquareAccumulator

#pragma region MatrixMatrixMul

void matrixMatrixMulSequential(float* A, float* B, float* C, int N, int M, int U) {
	for(int i = 0; i < M; i++) {
		for(int j = 0; j < N; j++) {
			float tmp = 0.0f;
			for(int k = 0; k < U; k++)
				tmp += A[i*U + k] * B[k*N + j];
			C[i*N + j] = tmp;
		}
	}
}

#if defined(_OPENMP)

void matrixMatrixMulOMP(float* A, float* B, float* C, int N, int M, int U) {
#pragma omp parallel shared(A,B,C,N,M,U) default(none)
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			float tmp = 0.0f;
			for (int k = 0; k < U; k++)
				tmp += A[i*U + k] * B[k*N + j];
			C[i*N + j] = tmp;
		}
	}
}
void matrixMatrixMulOMPTest() {
	printf("\nRunning Matrix Matrix Mul OMP\n");

	const unsigned int N = 3000;
	const unsigned int M = 3000;
	const unsigned int U = 3000;
	unsigned int mem_size_A = M*U * sizeof(float);
	unsigned int mem_size_B = U*N * sizeof(float);
	unsigned int mem_size_C = M*N * sizeof(float);
	float* h_A = (float*)malloc(mem_size_A);
	float* h_B = (float*)malloc(mem_size_B);
	float* h_C = (float*)malloc(mem_size_C);
	float* h_C_expected = (float*)malloc(mem_size_C);

	for (int i = 0; i<M; i++) {
		for (int j = 0; j<U; j++) {
			h_A[i*U + j] = rand() % 100;
		}
	}
	for (int i = 0; i<U; i++) {
		for (int j = 0; j<N; j++) {
			h_B[i*N + j] = rand() % 100;
		}
	}

	matrixMatrixMulOMP(h_A, h_B, h_C, M, N, U);
	matrixMatrixMulSequential(h_A, h_B, h_C, N, M, U);

	bool succeded;
	for (int i = 0; i<M; i++) {
		for (int j = 0; j<N; j++) {
			if (abs(h_C[i*N + j] - h_C_expected[i*N + j]) < 0.1f) {
				printf("\nFailed. Expected %f, was %f\n", h_C[i*N + j], h_C[i*N + j]);
				succeded = false;
				break;
			}
		}
		if (!succeded)
			break;
	}
	if (succeded)
		printf("\nCompleted successfully\n");


	free(h_A);
	free(h_B);
	free(h_C);
}

#else

void matrixMatrixMulGPU(
	unsigned int mem_size_A, 
	unsigned int mem_size_B, 
	unsigned int mem_size_C,
	float * h_A, 
	float * h_B, 
	float * h_C, 
	const unsigned int &M, 
	const unsigned int &N, 
	const unsigned int &U, 
	bool optimized)
{
	const unsigned int block_size = 512;
	const unsigned int T = 64;
	float* d_A;
	float* d_B;
	float* d_C;
	cudaMalloc((void**)&d_A, mem_size_A);
	cudaMalloc((void**)&d_B, mem_size_B);
	cudaMalloc((void**)&d_C, mem_size_C);
	// copy host memory to device
	cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

	unsigned long int elapsed;
	struct timeval t_start, t_end, t_diff;
	gettimeofday(&t_start, NULL);

	// run 
	matrixMatrixMul<T>(d_A, d_B, d_C, M, N, U, block_size, optimized);
	cudaThreadSynchronize();

	gettimeofday(&t_end, NULL);
	timeval_subtract(&t_diff, &t_end, &t_start);
	elapsed = (t_diff.tv_sec*1e6 + t_diff.tv_usec);
	double flops = 2.0 * M * N * U;
	double gigaFlops = (flops*1.0e-3f) / elapsed;

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
}

//  A, B and C have sizes MxU, UxN and MxN
void matrixMatrixMulGPUTest(bool optimized){
	if (optimized)
		printf("\nRunning Matrix Matrix Mul optimized\n");
	else
		printf("\nRunning Matrix Matrix Mul naive\n");

	const unsigned int N = 2500;
	const unsigned int M = 2500;
	const unsigned int U = 2500;
	unsigned int mem_size_A = M*U * sizeof(float);	
	unsigned int mem_size_B = U*N * sizeof(float);	
	unsigned int mem_size_C = M*N * sizeof(float);
	float* h_A = (float*)malloc(mem_size_A);
	float* h_B = (float*)malloc(mem_size_B);
	float* h_C = (float*)malloc(mem_size_C);
	float* h_C_expected = (float*)malloc(mem_size_C);

	for(int i = 0; i<M; i++) {
		for(int j = 0; j<U; j++){
			h_A[i*U + j] = rand() % 100;
		}
	}
	for(int i = 0; i<U; i++) {
		for(int j = 0; j<N; j++){
			h_B[i*N + j] = rand() % 100;
		}
	}

	matrixMatrixMulGPU(mem_size_A, mem_size_B, mem_size_C, h_A, h_B, h_C, M, N, U, optimized);
	matrixMatrixMulSequential(h_A, h_B, h_C, N, M, U);

	bool succeded;
	for (int i = 0; i<M; i++) {
		for (int j = 0; j<N; j++) {
			if (abs(h_C[i*N + j] - h_C_expected[i*N + j]) < 0.1f) {
				printf("\nFailed. Expected %f, was %f\n", h_C[i*N + j], h_C[i*N + j]);
				succeded = false;
				break;
			}
		}
		if (!succeded)
			break;
	}
	if (succeded)
		printf("\nCompleted successfully\n");
	

	free(h_A);
	free(h_B);
	free(h_C);
}

#endif

#pragma endregion MatrixMatrixMul


// ====== Runners

#if defined(_OPENMP)

void runOpenMPProgram() {
	matrixTransposeOMPTest();
	printf("\n==========================\n");
	matrixTransposeOMPTest();
}

#else 

void runGPUProgram() {
	matrixTransposeGPUTest(false);
	printf("\n==========================\n");
	matrixTransposeGPUTest(true);

	printf("\n==========================");
	printf("\n==========================\n");

	squareAccumulatorGPUTest(false);
	printf("\n==========================\n");
	squareAccumulatorGPUTest(true);

	printf("\n==========================");
	printf("\n==========================\n");

	matrixMatrixMulGPUTest(false);
	printf("\n==========================\n");
	matrixMatrixMulGPUTest(true);
}

#endif



int main(int argc, char** argv) {
#if defined(_OPENMP)
	runOpenMPProgram();
#else
	runGPUProgram();
#endif
}
