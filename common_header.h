#pragma once

#include <stdio.h>
#include <assert.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#pragma comment(lib, "cublas.lib")

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
 printf("Error at %s:%d\n",__FILE__,__LINE__);}} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
 printf("Error at %s:%d\n",__FILE__,__LINE__);}} while(0)

#define CUBLAS_CALL(x) do { if((x)!=CUBLAS_STATUS_SUCCESS) { \
 printf("Error at %s:%d\n",__FILE__,__LINE__);}} while(0)

// CUDA Runtime context (Singleton)
class CudaRuntimeContext
{
	static CudaRuntimeContext* instance;
	cublasHandle_t handle;

	CudaRuntimeContext() { cublasCreate(&handle); }
	~CudaRuntimeContext() { cublasDestroy(handle); }

public:
	static CudaRuntimeContext* getInstance()
	{
		if (!instance)
			instance = new CudaRuntimeContext();
		return instance;
	}
	cublasHandle_t cublasHandle() { return this->handle; }
};
CudaRuntimeContext* CudaRuntimeContext::instance = 0;

// Print matrix
void printMatrix(float* m, int rows, int cols)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			printf("%f\t", m[j * rows + i]);
		}
		printf("\n");
	}
	printf("\n");
}
