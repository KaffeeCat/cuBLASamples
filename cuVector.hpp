/////////////////////////////////////////
// Filename: cuMatrix.hpp
// ------------------------------------------------
// Purpose: Vector class optimized with cuBLAS
// Author: Wang Kang
// Date: 2022/2/10
/////////////////////////////////////////
#pragma once

#include "common_header.h"

template<typename T>
class cuVector
{
public:
	int len;
	T* data;

	cuVector(int len) :len(len), data(NULL)
	{
		assert(len > 0);
		CUDA_CALL(cudaMalloc(&data, len * sizeof(T)));
	}

	~cuVector()
	{
		len = 0;
		CUDA_CALL(cudaFree(data));
		data = NULL;
	}

	int size() const { return len * sizeof(T); }

	// Import data from host (PC memory)
	void copyFromHost(const T* host)
	{
		CUDA_CALL(cudaMemcpy(data, host, len * sizeof(T), cudaMemcpyHostToDevice));
	}

	// Export data to host (PC memory) 
	void copyToHost(T* host)
	{
		CUDA_CALL(cudaMemcpy(host, data, len * sizeof(T), cudaMemcpyDeviceToHost));
	}
};
