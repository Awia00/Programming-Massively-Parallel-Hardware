#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ScanHost.cu.h"

int scanIncTest(bool is_segmented) {
	const unsigned int num_threads = 8353455;
	const unsigned int block_size = 512;
	unsigned int mem_size = num_threads * sizeof(int);

	int* h_in = (int*)malloc(mem_size);
	int* h_out = (int*)malloc(mem_size);
	int* flags_h = (int*)malloc(num_threads * sizeof(int));

	int sgm_size = 123;
	{ // init segments and flags
		for (unsigned int i = 0; i<num_threads; i++) {
			h_in[i] = 1;
			flags_h[i] = (i % sgm_size == 0) ? 1 : 0;
		}
	}

	unsigned long int elapsed;
	struct timeval t_start, t_end, t_diff;
	gettimeofday(&t_start, NULL);


	{ // calling inclusive (segmented) scan
		int* d_in;
		int* d_out;
		int* flags_d;
		cudaMalloc((void**)&d_in, mem_size);
		cudaMalloc((void**)&d_out, mem_size);
		cudaMalloc((void**)&flags_d, num_threads * sizeof(int));

		// copy host memory to device
		cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(flags_d, flags_h, num_threads * sizeof(int), cudaMemcpyHostToDevice);

		// execute kernel
		if (is_segmented)
			sgmScanInc< Add<int>, int >(block_size, num_threads, d_in, flags_d, d_out);
		else
			scanInc< Add<int>, int >(block_size, num_threads, d_in, d_out);

		// copy host memory to device
		cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

		// cleanup memory
		cudaFree(d_in);
		cudaFree(d_out);
		cudaFree(flags_d);
	}

	gettimeofday(&t_end, NULL);
	timeval_subtract(&t_diff, &t_end, &t_start);
	elapsed = (t_diff.tv_sec*1e6 + t_diff.tv_usec);
	printf("Scan Inclusive GPU Kernel runs in: %lu microsecs\n", elapsed);

	// validation
	bool success = true;
	int  accum = 0;
	if (is_segmented) {
		for (int i = 0; i<num_threads; i++) {
			if (i % sgm_size == 0) accum = 0;
			accum += 1;

			if (accum != h_out[i]) {
				success = false;
				printf("Scan Inclusive Violation: %.1d should be %.1d\n", h_out[i], accum);
			}
		}
	}
	else {
		for (int i = 0; i<num_threads; i++) {
			accum += 1;

			if (accum != h_out[i]) {
				success = false;
				printf("Scan Inclusive Violation: %.1d should be %.1d\n", h_out[i], accum);
			}
		}
	}

	if (success) printf("\nScan Inclusive +   VALID RESULT!\n");
	else        printf("\nScan Inclusive + INVALID RESULT!\n");


	// cleanup memory
	free(h_in);
	free(h_out);
	free(flags_h);

	return 0;
}


int scanExcTest(bool is_segmented) {
	const unsigned int num_threads = 8353455;
	const unsigned int block_size = 512;
	unsigned int mem_size = num_threads * sizeof(int);

	int* h_in = (int*)malloc(mem_size);
	int* h_out = (int*)malloc(mem_size);
	int* flags_h = (int*)malloc(num_threads * sizeof(int));

	int sgm_size = 123;
	{ // init segments and flags
		for (unsigned int i = 0; i<num_threads; i++) {
			h_in[i] = 1;
			flags_h[i] = (i % sgm_size == 0) ? 1 : 0;
		}
	}

	unsigned long int elapsed;
	struct timeval t_start, t_end, t_diff;
	gettimeofday(&t_start, NULL);


	{ // calling exclusive (segmented) scan
		int* d_in;
		int* d_out;
		int* flags_d;
		cudaMalloc((void**)&d_in, mem_size);
		cudaMalloc((void**)&d_out, mem_size);
		cudaMalloc((void**)&flags_d, num_threads * sizeof(int));

		// copy host memory to device
		cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(flags_d, flags_h, num_threads * sizeof(int), cudaMemcpyHostToDevice);

		// execute kernel
		if (is_segmented)
			sgmScanExc< Add<int>, int >(block_size, num_threads, d_in, flags_d, d_out);
		else
			scanExc< Add<int>, int >(block_size, num_threads, d_in, d_out);

		// copy host memory to device
		cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

		// cleanup memory
		cudaFree(d_in);
		cudaFree(d_out);
		cudaFree(flags_d);
	}

	gettimeofday(&t_end, NULL);
	timeval_subtract(&t_diff, &t_end, &t_start);
	elapsed = (t_diff.tv_sec*1e6 + t_diff.tv_usec);
	printf("Scan Exclusive GPU Kernel runs in: %lu microsecs\n", elapsed);

	// validation
	bool success = true;
	int  accum = -1;
	if (is_segmented) {
		for (int i = 0; i<num_threads; i++) {
			if (i % sgm_size == 0) accum = -1;
			accum += 1;

			if (accum != h_out[i]) {
				success = false;
				printf("Sgm Scan Exclusive Violation: %.1d should be %.1d\n", h_out[i], accum);
				break;
			}
		}
	}
	else {
		for (int i = 0; i<num_threads; i++) {
			accum += 1;

			if (accum != h_out[i]) {
				success = false;
				printf("Scan Exclusive Violation: %.1d should be %.1d\n", h_out[i], accum);
				break;
			}
		}
	}

	if (success) printf("\nScan Exclusive +   VALID RESULT!\n");
	else        printf("\nScan Exclusive + INVALID RESULT!\n");


	// cleanup memory
	free(h_in);
	free(h_out);
	free(flags_h);

	return 0;
}

int msspCpu(int* arr, int arr_size) {
	int maxpre = 0, maxseg = 0;
	for (int i = 0; i < arr_size; i++) {
		maxpre = max(0, (maxpre + arr[i]));
		maxseg = max(maxpre, maxseg);
	}
	return maxseg;
}

int msspTest() {
	const unsigned int num_threads = 8353455;
	const unsigned int block_size = 512;
	unsigned int mem_size = num_threads * sizeof(int);
	unsigned int mem_size_int4 = num_threads * sizeof(MyInt4);

	int* h_in = (int*)malloc(mem_size);
	MyInt4* h_out = (MyInt4*)malloc(mem_size_int4);

	for (unsigned int i = 0; i<num_threads; i++) {
		h_in[i] = (rand() % 100) - 49;
	}

	unsigned long int elapsed;
	struct timeval t_start, t_end, t_diff;
	gettimeofday(&t_start, NULL);


	{ // calling exclusive (segmented) scan
		int* d_in;
		MyInt4* d_out;
		cudaMalloc((void**)&d_in, mem_size);
		cudaMalloc((void**)&d_out, mem_size_int4);

		// copy host memory to device
		cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);

		// execute kernel
		mssp(block_size, num_threads, d_in, d_out);

		// copy host memory to device
		cudaMemcpy(h_out, d_out, mem_size_int4, cudaMemcpyDeviceToHost);

		// cleanup memory
		cudaFree(d_in);
		cudaFree(d_out);
	}

	gettimeofday(&t_end, NULL);
	timeval_subtract(&t_diff, &t_end, &t_start);
	elapsed = (t_diff.tv_sec*1e6 + t_diff.tv_usec);
	printf("MSSP GPU Kernel runs in: %lu microsecs\n", elapsed);

	// validate
	int cpuResult = msspCpu(h_in, num_threads);

	if (cpuResult == h_out[num_threads - 1].x)	printf("\nMSSP +   VALID RESULT!\n");
	else										printf("\nMSSP + INVALID RESULT!\n");

	// cleanup memory
	free(h_in);
	free(h_out);

	return 0;
}

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

	unsigned long int elapsed;
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
		sp_matrix_multiply(block_size, matrix_size, vector_size, d_mat_inds, d_mat_val, d_vct, d_flags, d_out);

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
	printf("MSSP GPU Kernel runs in: %lu microsecs\n", elapsed);

	// validate
	float* h_test = (float*)malloc(mem_size_vct);
	spMatrixVctMultiply(h_mat_val, h_mat_inds, h_vct, matrix_height, h_shp, h_test);

	bool success = true;
	for (int i = 0; i < matrix_height; i++) {
		if (abs(h_test[i] - h_out[i]) > 0.0001f) {
			success = false;
			printf("\nFailed at: h_test[%d]: %f, h_out[%d]=%f\n", i, h_test[i], i, h_out[i]);
			break;
		}
	}

	if (success) printf("\nSparse Matrix Vector Multiply +   VALID RESULT!\n");
	else        printf("\nSparse Matrix Vector Multiply + INVALID RESULT!\n");

	free(h_mat_val);
	free(h_vct);
	free(h_out);
	free(h_mat_inds);
	free(h_flags);
}

int main(int argc, char** argv) {
	scanIncTest(true);
	scanIncTest(true);
	scanIncTest(false);

	scanExcTest(true);
	scanExcTest(true);
	scanExcTest(false);

	msspTest();
	msspTest();

	spMatrixVctTest();
	spMatrixVctTest();
}