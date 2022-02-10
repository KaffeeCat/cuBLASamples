/////////////////////////////////////////
// Filename: cuMatrix.hpp
// ------------------------------------------------
// Purpose: Matrix class optimized with cuBLAS
// Author: Wang Kang
// Date: 2022/2/10
/////////////////////////////////////////
#pragma once

#include "cuVector.hpp"

class cuMatrix32f : public cuVector<float>
{
public:
	int rows;
	int cols;

	cuMatrix32f(int rows, int cols) :rows(rows), cols(cols), cuVector(rows* cols) { assert(rows > 0 && cols > 0); }
	~cuMatrix32f() { rows = cols = 0; }

	// Multiply A(class itself) and B on GPU and save the result in C
	// C(m,n) = A(m,k) * B(k,n)
	void matmul(const cuMatrix32f & B, cuMatrix32f & C) const
	{
		const cuMatrix32f &A = *this;

		assert(C.rows == A.rows);
		assert(C.cols == B.cols);
		assert(A.cols == B.rows);

		int m = C.rows, n = C.cols, k = A.cols;
		int lda = A.cols, ldb = B.cols, ldc = C.rows;
		const float alf = 1, bet = 0;
		const float* alpha = &alf, * beta = &bet;

		cublasHandle_t handle = CudaRuntimeContext::getInstance()->cublasHandle();
		// CUBLAS_OP_N : the non-transpose operation is selected ---> leading dimention is row
		// CUBLAS_OP_T : the transpose operation is selected ---> leading dimention is column
		CUBLAS_CALL(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, alpha, A.data, lda, B.data, ldb, beta, C.data, ldc));
	}
};
