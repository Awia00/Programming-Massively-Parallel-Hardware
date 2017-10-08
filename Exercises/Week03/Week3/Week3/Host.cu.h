#ifndef SCAN_HOST
#define SCAN_HOST

#include "Kernels.cu.h"
#include "device_launch_parameters.h"


/**
 * block_size is the size of the cuda block (must be a multiple 
 *                of 32 less than 1025)
 * d_size     is the size of both the input and output arrays.
 * d_in       is the device array; it is supposably
 *                allocated and holds valid values (input).
 * d_out      is the output GPU array -- if you want 
 *            its data on CPU needs to copy it back to host.
 *
 * OP         class denotes the associative binary operator 
 *                and should have an implementation similar to 
 *                `class Add' in ScanUtil.cu, i.e., exporting
 *                `identity' and `apply' functions.
 * T          denotes the type on which OP operates, 
 *                e.g., float or int. 
 */
template<class OP, class T>
void scanInc(    unsigned int  block_size,
                 unsigned long d_size, 
                 T*            d_in,  // device
                 T*            d_out  // device
) {
    unsigned int num_blocks;
    unsigned int sh_mem_size = block_size * 32; //sizeof(T);

    num_blocks = ( (d_size % block_size) == 0) ?
                    d_size / block_size     :
                    d_size / block_size + 1 ;

    scanIncKernel<OP,T><<< num_blocks, block_size, sh_mem_size >>>(d_in, d_out, d_size);
    cudaThreadSynchronize();
    
    if (block_size >= d_size) { return; }

    /**********************/
    /*** Recursive Case ***/
    /**********************/

    //   1. allocate new device input & output array of size num_blocks
    T *d_rec_in, *d_rec_out;
    cudaMalloc((void**)&d_rec_in , num_blocks*sizeof(T));
    cudaMalloc((void**)&d_rec_out, num_blocks*sizeof(T));

    unsigned int num_blocks_rec = ( (num_blocks % block_size) == 0 ) ?
                                  num_blocks / block_size     :
                                  num_blocks / block_size + 1 ; 

    //   2. copy in the end-of-block results of the previous scan 
    copyEndOfBlockKernel<T><<< num_blocks_rec, block_size >>>(d_out, d_rec_in, num_blocks);
    cudaThreadSynchronize();

    //   3. scan recursively the last elements of each CUDA block
    scanInc<OP,T>( block_size, num_blocks, d_rec_in, d_rec_out );

    //   4. distribute the the corresponding element of the 
    //      recursively scanned data to all elements of the
    //      corresponding original block
    distributeEndBlock<OP,T><<< num_blocks, block_size >>>(d_rec_out, d_out, d_size);
    cudaThreadSynchronize();

    //   5. clean up
    cudaFree(d_rec_in );
    cudaFree(d_rec_out);
}

template<class OP, class T>
void scanExc(    unsigned int  block_size,
                 unsigned long d_size, 
                 T*            d_in,  // device
                 T*            d_out  // device
) {
	T *d_inc;
	cudaMalloc((void**)&d_inc, d_size * sizeof(T));

    scanInc<OP,T>(block_size, d_size, d_in, d_inc);
	cudaThreadSynchronize();

	unsigned int num_blocks = ( (d_size % block_size) == 0) ?
                    d_size / block_size     :
                    d_size / block_size + 1 ;
    
    shiftRightByOne<T><<< num_blocks, block_size >>>(d_inc, d_out, OP::identity(), d_size);
	cudaThreadSynchronize();
	cudaFree(d_inc);
}


/**
 * block_size is the size of the cuda block (must be a multiple 
 *                of 32 less than 1025)
 * d_size     is the size of both the input and output arrays.
 * d_in       is the device array; it is supposably
 *                allocated and holds valid values (input).
 * flags      is the flag array, in which !=0 indicates 
 *                start of a segment.
 * d_out      is the output GPU array -- if you want 
 *            its data on CPU you need to copy it back to host.
 *
 * OP         class denotes the associative binary operator 
 *                and should have an implementation similar to 
 *                `class Add' in ScanUtil.cu, i.e., exporting
 *                `identity' and `apply' functions.
 * T          denotes the type on which OP operates, 
 *                e.g., float or int. 
 */
template<class OP, class T>
void sgmScanInc( const unsigned int  block_size,
                 const unsigned long d_size,
                 T*            d_in,  //device
                 int*          flags, //device
                 T*            d_out  //device
) {
    unsigned int num_blocks;
    //unsigned int val_sh_size = block_size * sizeof(T  );
    unsigned int flg_sh_size = block_size * sizeof(int);

    num_blocks = ( (d_size % block_size) == 0) ?
                    d_size / block_size     :
                    d_size / block_size + 1 ;

    T     *d_rec_in;
    int   *f_rec_in;
    cudaMalloc((void**)&d_rec_in, num_blocks*sizeof(T  ));
    cudaMalloc((void**)&f_rec_in, num_blocks*sizeof(int));

    sgmScanIncKernel<OP,T> <<< num_blocks, block_size, 32*block_size >>>
                    (d_in, flags, d_out, f_rec_in, d_rec_in, d_size);
    cudaThreadSynchronize();
    //cudaError_t err = cudaThreadSynchronize();
    //if( err != cudaSuccess)
    //    printf("cudaThreadSynchronize error: %s\n", cudaGetErrorString(err));

    if (block_size >= d_size) { cudaFree(d_rec_in); cudaFree(f_rec_in); return; }

    //   1. allocate new device input & output array of size num_blocks
    T   *d_rec_out;
    int *f_inds;
    cudaMalloc((void**)&d_rec_out, num_blocks*sizeof(T   ));
    cudaMalloc((void**)&f_inds,    d_size    *sizeof(int ));

    //   2. recursive segmented scan on the last elements of each CUDA block
    sgmScanInc<OP,T>
                ( block_size, num_blocks, d_rec_in, f_rec_in, d_rec_out );

    //   3. create an index array that is non-zero for all elements
    //      that correspond to an open segment that crosses two blocks,
    //      and different than zero otherwise. This is implemented
    //      as a CUDA-block level inclusive scan on the flag array,
    //      i.e., the segment that start the block has zero-flags,
    //      which will be preserved by the inclusive scan. 
    scanIncKernel<Add<int>,int> <<< num_blocks, block_size, flg_sh_size >>>
                ( flags, f_inds, d_size );

    //   4. finally, accumulate the recursive result of segmented scan
    //      to the elements from the first segment of each block (if 
    //      segment is open).
    sgmDistributeEndBlock <OP,T> <<< num_blocks, block_size >>>
                ( d_rec_out, d_out, f_inds, d_size );
    cudaThreadSynchronize();

    //   5. clean up
    cudaFree(d_rec_in );
    cudaFree(d_rec_out);
    cudaFree(f_rec_in );
    cudaFree(f_inds   );
}

template<class OP, class T>
void sgmScanExc( const unsigned int  block_size,
                 const unsigned long d_size,
                 T*            d_in,  //device
                 int*          flags, //device
                 T*            d_out  //device
) {
	T *d_inc;
	cudaMalloc((void**)&d_inc, d_size * sizeof(T));

	sgmScanInc<OP, T>(block_size, d_size, d_in, flags, d_inc);
	cudaThreadSynchronize();

	unsigned int num_blocks = ( (d_size % block_size) == 0) ?
                    d_size / block_size     :
                    d_size / block_size + 1 ;

    sgmShiftRightByOne<T><<< num_blocks, block_size >>>(d_inc, flags, d_out, OP::identity(), d_size);
	cudaThreadSynchronize();
	cudaFree(d_inc);
}

void sp_matrix_vector_multiply(const unsigned int  block_size,
	const unsigned long d_tot_size,
	const unsigned long d_vct_len,
	int*				d_mat_inds, 
	float*				d_mat_vals,
	float*				d_vct,
	int*				d_flags,
	float*				d_out	//device
) {
	float *d_pairs;
	float *d_tmp_scan;
	int *d_tmp_inds;
	cudaMalloc((void**)&d_pairs, d_tot_size * sizeof(float));
	cudaMalloc((void**)&d_tmp_scan, d_tot_size * sizeof(float));
	cudaMalloc((void**)&d_tmp_inds, d_tot_size * sizeof(int));

	unsigned int num_blocks;
	num_blocks = ((d_tot_size % block_size) == 0) ?
		d_tot_size / block_size :
		d_tot_size / block_size + 1;

	spMatVctMult_pairs<<<num_blocks, block_size>>> (d_mat_inds, d_mat_vals, d_vct, d_tot_size, d_pairs);

	sgmScanInc<Add<float>, float>(block_size, d_tot_size, d_pairs, d_flags, d_tmp_scan);
	scanInc<Add<int>,int>(block_size, d_tot_size, d_flags, d_tmp_inds);
	
	write_lastSgmElem<<<num_blocks, block_size>>> (d_tmp_scan, d_tmp_inds, d_flags, d_tot_size, d_out);
	
	cudaFree(d_pairs);
	cudaFree(d_tmp_inds);
}

void matrix_transpose(unsigned int blocksize
    const unsigned long m,
	const unsigned long n,
	float*				d_matrix,
    float*				d_out,
    bool                optimized
) {
    int dimy = ceil( ((float)M) / T );
    int dimx = ceil( ((float)N) / T );
    dim3 block(T,T,1), grid(dimx,dimy,1);

    if (optimized)
        matrix_transpose<<<grid, block>>> (d_matrix, d_out, m, n);
    else 
        matrix_transpose_naive<<<grid, block>>>(d_matrix, d_out, m, n);
}


void list_sq_accumulator(unsigned int blocksize
    const unsigned long N,
	float*				d_in,
    float*				d_out, 
    bool                optimized
) {
    unsigned int num_blocks = ( (N % block_size) == 0) ?
    N / block_size     :
    N / block_size + 1 ;

    int size = N*64;
    float *d_in_trans;
    float *d_out_trans;
	cudaMalloc((void**)&d_in_trans, size * sizeof(float));
	cudaMalloc((void**)&d_out_trans, size * sizeof(float));

    if(optimized) {
        matrix_transpose(blocksize, n, 64, d_in, d_in_trans, true);
        listSqrtAccumulatorTranspose<<<num_blocks, blocksize>>>(d_in_trans, d_out_trans, N);
        matrix_transpose(blocksize, n, 64, d_out_trans, d_out, true);
    } else {
        listSqrtAccumulator<<<num_blocks, blocksize>>>(d_in, d_out, N);
    }
}

void matrix_matrix_mul(unsigned int blocksize
    const unsigned long M,
	const unsigned long N,
	const unsigned long U,
	float*				d_A,
	float*				d_B,
    float*				d_C
) {
    int dimy = ceil( ((float)M) / T );
    int dimx = ceil( ((float)N) / T );
    dim3 block(T,T,1), grid(dimx,dimy,1);

    matMatMul<T><<<grid, block>>>(d_A, d_B, d_C, M, N, U);
}
#endif //SCAN_HOST


