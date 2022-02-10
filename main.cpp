/////////////////////////////////////////
// Filename: main.cpp
// ------------------------------------------------
// Purpose: How to use cuMatrix to do matrix operations optimized with cuBLAS
// Author: Wang Kang
// Date: 2022/2/10
/////////////////////////////////////////
#include "cuMatrix.hpp"
#include "TimeMeter.hpp"

int main()
{
	// Initialize cuBLAS handle
	cublasHandle_t handle = CudaRuntimeContext::getInstance()->cublasHandle();
	TimeMeter timemeter;

	// Define matrix A and B
	float a[6] = { 1, 2, 3, 4, 5, 6 };
	float b[12] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	printf(">> Matrix A\n");
	printMatrix(a, 2, 3);
	printf(">> Matrix B\n");
	printMatrix(b, 3, 4);

	// Copy matrix to device
	cuMatrix32f A(2, 3), B(3, 4);
	A.copyFromHost(a);
	B.copyFromHost(b);

	// Do matrix multiplication
	timemeter.start();
	cuMatrix32f C(2, 4);
	for (int i = 0; i < 10000; i++)
	{
		A.matmul(B, C);
	}
	printf("Time usage : %.2f ms\n\n", timemeter.stop());

	// Copy matrix to host
	float c[8];
	C.copyToHost(c);
	printf(">> Matrix C\n");
	printMatrix(c, 2, 4);

	return 0;
}