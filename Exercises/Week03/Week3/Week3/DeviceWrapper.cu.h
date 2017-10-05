#ifndef DEVICE_WRAPPER
#define DEVICE_WRAPPER
#include "device_launch_parameters.h"
class Matrix {
	public:
		Matrix() {}
		virtual ~Matrix() {}
		virtual void sp_matrix_vector_multiply(const unsigned int  block_size,
			const unsigned long d_tot_size,
			const unsigned long d_vct_len,
			int*				d_mat_inds,
			float*				d_mat_vals,
			float*				d_vct,
			int*				d_flags,
			float*				d_out) = 0;
};
#endif