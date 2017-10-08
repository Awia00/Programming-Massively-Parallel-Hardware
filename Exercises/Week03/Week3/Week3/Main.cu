#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "Host.cu.h"

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#include "TimeOfDay.h" // Only on windows
#else
#include <sys/time.h> // Only on linux
#endif


void spMatrixVctMultiply(float* mat_val, int* mat_inds, float* vct, int num_rows, int* shp, float* res) {
	int offset = 0;
	for (int i = 0; i < num_rows; i++) {
		int sum = 0, row_len = shp[i];
		for (int j = 0; j < row_len; j++) {
			sum += mat_val[j + offset] * vct[mat_inds[j + offset]];
		}
		offset += row_len;
		res[i] = sum;
	}
}

void spMatrixVctTest() {
	const unsigned int block_size = 512;
	const unsigned int matrix_height = 5733;
	const unsigned int matrix_width = 5733;
	const unsigned int matrix_size = matrix_height * matrix_width;
	const unsigned int vector_size = matrix_height;

	unsigned int mem_size_mat_vals	= matrix_size * sizeof(float);
	unsigned int mem_size_mat_inds	= matrix_size * sizeof(int);
	unsigned int mem_size_shp		= matrix_height * sizeof(int);
	unsigned int mem_size_vct		= vector_size * sizeof(float);

	float* h_mat_val = (float*)malloc(mem_size_mat_vals);
	float* h_vct = (float*)malloc(mem_size_vct);
	float* h_out = (float*)malloc(mem_size_vct);
	int* h_mat_inds = (int*)calloc(matrix_size, sizeof(int));
	int* h_shp = (int*)calloc(matrix_height, sizeof(int));
	int* h_flags = (int*)calloc(matrix_size, sizeof(int));

	// generate matrix. Did not generate a matrix with any sparse entries, but code shuld be able to handle it.
	for (unsigned int i = 0; i < matrix_height; i++) {
		int nonZeros = 0;
		for (unsigned int j = 0; j < matrix_width; j++) {
			h_mat_val[i*matrix_height + j] = rand() % 100 + 1; // 1-100
			h_mat_inds[i*matrix_height + j] = j; // 1-100
			nonZeros++;
		}
		h_shp[i] = nonZeros;
		h_vct[i] = rand() % 100 + 1;
		h_flags[i*matrix_width] = 1;
	}

	unsigned long int elapsed, elapsedCPU;
	struct timeval t_start, t_end, t_diff;
	gettimeofday(&t_start, NULL);

	{ // calling exclusive (segmented) scan
		float* d_mat_val;
		float* d_vct;
		float* d_out;
		int* d_mat_inds;
		int* d_flags;
		
		cudaMalloc((void**)&d_mat_val, mem_size_mat_vals);
		cudaMalloc((void**)&d_vct, mem_size_vct);
		cudaMalloc((void**)&d_out, mem_size_vct);
		cudaMalloc((void**)&d_mat_inds, mem_size_mat_inds);
		cudaMalloc((void**)&d_flags, mem_size_mat_inds);

		// copy host memory to device
		cudaMemcpy(d_mat_val, h_mat_val, mem_size_mat_vals, cudaMemcpyHostToDevice);
		cudaMemcpy(d_vct, h_vct, mem_size_vct, cudaMemcpyHostToDevice);
		cudaMemcpy(d_mat_inds, h_mat_inds, mem_size_mat_inds, cudaMemcpyHostToDevice);
		cudaMemcpy(d_flags, h_flags, mem_size_mat_inds, cudaMemcpyHostToDevice);

		// execute kernel
		sp_matrix_vector_multiply(block_size, matrix_size, vector_size, d_mat_inds, d_mat_val, d_vct, d_flags, d_out);

		// copy host memory to device
		cudaMemcpy(h_out, d_out, mem_size_vct, cudaMemcpyDeviceToHost);

		// cleanup memory
		cudaFree(d_mat_val);
		cudaFree(d_vct);
		cudaFree(d_mat_inds);
		cudaFree(d_flags);
		cudaFree(d_out);
	}

	gettimeofday(&t_end, NULL);
	timeval_subtract(&t_diff, &t_end, &t_start);
	elapsed = (t_diff.tv_sec*1e6 + t_diff.tv_usec);
	printf("SP MV MUL GPU Kernel runs in: \t%lu microsecs\n", elapsed);

	// validate
	float* h_test = (float*)malloc(mem_size_vct);
	
	gettimeofday(&t_start, NULL);
	spMatrixVctMultiply(h_mat_val, h_mat_inds, h_vct, matrix_height, h_shp, h_test);
	gettimeofday(&t_end, NULL);
	timeval_subtract(&t_diff, &t_end, &t_start);
	elapsedCPU = (t_diff.tv_sec*1e6 + t_diff.tv_usec);
	printf("SP MV MUL CPU runs in: \t\t%lu microsecs\n", elapsedCPU);

	elapsed = (t_diff.tv_sec*1e6 + t_diff.tv_usec);


	bool success = true;
	for (int i = 0; i < matrix_height; i++) {
		if (abs(h_test[i] - h_out[i]) > 0.0001f) {
			success = false;
			printf("Failed at: h_test[%d]: %f, h_out[%d]=%f\n", i, h_test[i], i, h_out[i]);
			break;
		}
	}

	if (success) printf("Sparse Matrix Vector Multiply +   VALID RESULT!\n");
	else        printf("Sparse Matrix Vector Multiply + INVALID RESULT!\n");

	free(h_mat_val);
	free(h_vct);
	free(h_out);
	free(h_mat_inds);
	free(h_flags);
}

void matrixTranspose(float* matrix, float* outMatrix, int n, int m) {
	for(int i = 0; i < m; i++) {
		for(int j = 0; j < n; j++){
			outMatrix[j][i] = matrix[i][j];
		}
	}
}

void matrixTransposeTest(bool optimized) {
	const unsigned int block_size = 512;
	const unsigned int N = 1000;
	const unsigned int M = 1000;
	unsigned int mem_size_A = M*N * sizeof(float);	
	float* h_A = (float*)malloc(mem_size_A);
	float* h_A_out = (float*)malloc(mem_size_A);

	for(int i = 0; i<M; i++) {
		for(int j = 0; j<N; j++){
			A[i*M + j] = rand() % 100;
		}
	}
	
	unsigned long int elapsed;
	struct timeval t_start, t_end, t_diff;

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
	matrix_transpose(d_A, d_A_out, M, N, optimized);
	__syncThreads();
	
	gettimeofday(&t_end, NULL);
	timeval_subtract(&t_diff, &t_end,&t_start);
	if(optimized) {
		printf("M-transpose optimized time\t %d", elapsedTime);
	} else { 
		printf("M-transpose naive time\t %d", elapsedTime);
	}

	cudaMemcpy(h_A_out, d_A_out, mem_size_A, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_A_out);
}

void squareAccumulatorTest(bool optimized) {
	const unsigned int block_size = 512;
	const unsigned int N = 1000;
	const unsigned int M = 64;
	unsigned int mem_size = M*N * sizeof(float);	
	float* h_A = (float*)malloc(mem_size);
	float* h_B = (float*)malloc(mem_size);

	for(int i = 0; i<M; i++) {
		for(int j = 0; j<N; j++){
			A[i*M + j] = rand() % 100;
		}
	}
	
	unsigned long int elapsed;
	struct timeval t_start, t_end, t_diff;

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
	square_accumulator(N, d_A, d_B, optimized);
	__syncThreads();
	
	gettimeofday(&t_end, NULL);
	timeval_subtract(&t_diff, &t_end,&t_start);

	if(optimized)
		printf("Square Accumulator optimized time\t %d", elapsedTime);	
	else
		printf("Square Accumulator naive time\t %d", elapsedTime);	
	
	cudaMemcpy(h_B, d_B, mem_size, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
}

void denseMatrixMatrixMul(float* A, float* B, float* C, int N, int M, int U) {
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
	const unsigned int block_size = 512;
	const unsigned int N = 1000;
	const unsigned int M = 1000;
	const unsigned int U = 1000;
	unsigned int mem_size_A = M*U * sizeof(float);	
	unsigned int mem_size_B = U*N * sizeof(float);	
	unsigned int mem_size_C = M*N * sizeof(float);
	float* h_A = (float*)malloc(mem_size_A);
	float* h_B = (float*)malloc(mem_size_B);
	float* h_C = (float*)malloc(mem_size_C);

	for(int i = 0; i<M; i++) {
		for(int j = 0; j<U; j++){
			A[i*M + j] = rand() % 100;
		}
	}
	for(int i = 0; i<U; i++) {
		for(int j = 0; j<N; j++){
			B[i*U + j] = rand() % 100;
		}
	}
	for(int i = 0; i<M; i++) {
		for(int j = 0; j<N; j++){
			C[i*M + j] = rand() % 100;
		}
	}
	
	unsigned long int elapsed;
	struct timeval t_start, t_end, t_diff;

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
	matrix_matrix_mul(d_A, d_B, d_C, M, N, U, optimized);
	__syncThreads();
	
	gettimeofday(&t_end, NULL);
	timeval_subtract(&t_diff, &t_end,&t_start);
	elapsed=(t_diff.tv_sec*1e6 + t_diff.tv_usec);
	double flops = 2.0 * M * N * U;
	double gigaFlops=(flops*1.0e-3f) /elapsed;
	if(optimized) {
		printf("MMM optimized time\t %d", elapsedTime);		
		printf("MMM gigaFlops optimized %d", gigaFlops);
	} else { 
		printf("MMM naive time\t %d", elapsedTime);				
		printf("MMM gigaFlops naive %d", gigaFlops);
	}

	cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

float gflops(){
	unsigned long int elapsed;
	struct timeval t_start,t_end,t_diff;
	gettimeofday(&t_start, NULL);
	
	gettimeofday(&t_end, NULL);
	timeval_subtract(&t_diff, &t_end,&t_start);
	elapsed=(t_diff.tv_sec*1e6 + t_diff.tv_usec);
	double flops = 2.0 * M * N * U;
	double gigaFlops=(flops*1.0e-3f) /elapsed;
}
int main(int argc, char** argv) {
	printf("\n");

	spMatrixVctTest();
	printf("\n==========================\n");
	spMatrixVctTest();
}

