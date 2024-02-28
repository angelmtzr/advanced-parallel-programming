
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdint>
#include <iostream>
#include <ctime>
#include <cstdio>

cudaError_t mMultWithCuda(const uint8_t* a, const uint8_t* b, uint16_t* c,
	const uint8_t M, const uint8_t N, const uint8_t P);

void mMultCPU(const uint8_t* a, const uint8_t* b, uint16_t* c,
	const uint8_t M, const uint8_t N, const uint8_t P)
{
	for (uint8_t i = 0; i < M; i++) {
		for (uint8_t j = 0; j < P; j++) {
			uint16_t sum = 0;
			for (uint8_t k = 0; k < N; k++) {
				sum += static_cast<uint16_t>(a[i * N + k]) * static_cast<uint16_t>(b[k * P + j]);
			}
			c[i * P + j] = sum;
		}
	}
}

void printMatrix(const uint16_t* matrix, const uint8_t rows, const uint8_t columns) {
	for (uint8_t i = 0; i < rows; i++) {
		for (uint8_t j = 0; j < columns; j++) {
			std::cout << matrix[i * columns + j] << "\t";
		}
		std::cout << std::endl;
	}
}

void printMatrix(const uint8_t* matrix, const uint8_t rows, const uint8_t columns) {
	for (uint8_t i = 0; i < rows; ++i) {
		for (uint8_t j = 0; j < columns; ++j) {
			std::cout << static_cast<int>(matrix[i * columns + j]) << "\t";
		}
		std::cout << std::endl;
	}
}

__global__ void mMultKernel(const uint8_t* a, const uint8_t* b, uint16_t* c,
	const uint8_t M, const uint8_t N, const uint8_t P)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < M && col < P) {
		uint16_t sum = 0;
		for (int i = 0; i < N; i++) 
		{
			sum += static_cast<uint16_t>(a[row * N + i]) * static_cast<uint16_t>(b[i * P + col]);
		}
		c[row * P + col] = sum;
	}
}

int main()
{
	// Matrix dimensions
	const uint8_t M = 4;
	const uint8_t N = 4;
	const uint8_t P = 4;

	// Host matrices
	const uint8_t h_a[M * N] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
	const uint8_t h_b[N * P] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
	uint16_t h_c[M * P];

	// Matrix multiplication in CPU
	clock_t start_CPU = clock();
	mMultCPU(h_a, h_b, h_c, M, N, P);
	clock_t end_CPU = clock();
	double time_CPU = double(end_CPU - start_CPU) / CLOCKS_PER_SEC;

	// Output the matrix operation
	std::cout << "CPU matrix multiplication:" << std::endl;
	printMatrix(h_a, M, N); std::cout << std::endl;
	printMatrix(h_b, N, P); std::cout << std::endl;
	printMatrix(h_c, M, P);
	std::cout << "Execution time: " << time_CPU << " seconds" << std::endl << std::endl;

	// Matrix multiplication in GPU (parallel)
	clock_t start_GPU = clock();
	cudaError_t cudaStatus = mMultWithCuda(h_a, h_b, h_c, M, N, P);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "mMultWithCuda failed!");
		return EXIT_FAILURE;
	}
	clock_t end_GPU = clock();
	double time_GPU = double(end_CPU - start_CPU) / CLOCKS_PER_SEC;

	// Output the matrix operation
	std::cout << "GPU matrix multiplication:" << std::endl;
	printMatrix(h_a, M, N); std::cout << std::endl;
	printMatrix(h_b, N, P); std::cout << std::endl;
	printMatrix(h_c, M, P);
	std::cout << "Execution time: " << time_GPU << " seconds" << std::endl;


	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

// Helper function for using CUDA to multiplicate matrices in parallel.
cudaError_t mMultWithCuda(const uint8_t* a, const uint8_t* b, uint16_t* c,
	const uint8_t M, const uint8_t N, const uint8_t P)
{
	// Device matrices
	uint8_t* d_a, * d_b;
	uint16_t* d_c;

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&d_a, M * N * sizeof(uint8_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_b, N * P * sizeof(uint8_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_c, M * P * sizeof(uint16_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy matrices A and B from host to device
	cudaStatus = cudaMemcpy(d_a, a, M * N * sizeof(uint8_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_b, b, N * P * sizeof(uint8_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Define dynamic grid and block dimensions
	dim3 dimBlock(32, 32, 1);
	// Ex.((4+32-1) / 32), (4+32-1)/32, 1)
	// (1, 1, 1)
	dim3 dimGrid((P + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y, 1);

	// Launch a kernel on the GPU with one thread for each element.
	mMultKernel << <dimGrid, dimBlock >> > (d_a, d_b, d_c, M, N, P);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "mMultKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching mMultKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, d_c, M * P * sizeof(uint16_t), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return cudaStatus;
}
