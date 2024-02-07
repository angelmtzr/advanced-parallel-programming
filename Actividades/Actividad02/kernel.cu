
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void print_all_idx(const int* a, const int* b, int* c) {
	printf("[DEVICE] (threadIdx.x, blockIdx.x, gridDim.x) -> (%d, %d, %d)\n",
		threadIdx.x, blockIdx.x, gridDim.x);
	printf("[DEVICE] (threadIdx.y, blockIdx.y, gridDim.y) -> (%d, %d, %d)\n",
		threadIdx.y, blockIdx.y, gridDim.y);
	printf("[DEVICE] (threadIdx.z, blockIdx.z, gridDim.z) -> (%d, %d, %d)\n",
		threadIdx.z, blockIdx.z, gridDim.z);
}

int main()
{
	dim3 blockSize(4, 4, 4);
	dim3 gridSize(2, 2, 2);

	const int arraySize = 10000;
	const int arrayBytes = arraySize * sizeof(int);

	// Host arrays
	int* h_a = new int[arraySize];
	int* h_b = new int[arraySize];
	int* h_c = new int[arraySize];

	// Device arrays
	int* d_a, * d_b, * d_c;
	cudaMalloc((void**)&d_a, arrayBytes);
	cudaMalloc((void**)&d_b, arrayBytes);
	cudaMalloc((void**)&d_c, arrayBytes);

	// Copy data from HOST to DEVICE
	cudaMemcpy(d_a, h_a, arrayBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, arrayBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, h_c, arrayBytes, cudaMemcpyHostToDevice);

	// Launch to kernel
	print_all_idx << <gridSize, blockSize >> > (d_a, d_b, d_c);

	// Copy result from DEVICE to HOST
	cudaMemcpy(h_a, d_a, arrayBytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_b, d_b, arrayBytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_c, d_c, arrayBytes, cudaMemcpyDeviceToHost);

	cudaDeviceReset();

	// Free memory
	delete[] h_a;
	delete[] h_b;
	delete[] h_c;
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return EXIT_SUCCESS;
}
