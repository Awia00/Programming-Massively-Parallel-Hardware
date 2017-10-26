void MatrixTransposeGPU(unsigned int mem_size_A, float * h_A, float * h_A_out, const unsigned int &T, const unsigned int &block_size, const unsigned int &M, const unsigned int &N, bool optimized);

void squareAccumulatorGPU(unsigned int mem_size, float * h_A, float * h_B, const unsigned int &T, const unsigned int &block_size, const unsigned int &N, bool optimized);

void matrixMatrixMulGPU(unsigned int mem_size_A, unsigned int mem_size_B, unsigned int mem_size_C, float * h_A, float * h_B, float * h_C, const unsigned int &T, const unsigned int &M, const unsigned int &N, const unsigned int &U, const unsigned int &block_size, bool optimized);
