#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define GPUErrorAssertion(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void unrolling2_kernel(int* input, int* temp, int size) {
    int tid = threadIdx.x;
    int BLOCK_OFFSET = blockIdx.x * blockDim.x * 2;
    int index = BLOCK_OFFSET + tid;

    int* i_data = input + BLOCK_OFFSET;

    if ((index + blockDim.x) < size) {
        input[index] += input[index + blockDim.x];
    }

    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            i_data[tid] += i_data[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        temp[blockIdx.x] = i_data[0];
    }
}

__global__ void transpose_kernel(int* input, int* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index_in = y * width + x;
        int index_out = x * height + y;
        output[index_out] = input[index_in];
    }
}

int main() {
    int data_size = 1 << 10;
    int byte_size = data_size * sizeof(int);
    int block_size = 32;
    int parallel_reduction = 2;

    int* host_input, * host_ref;
    host_input = (int*)malloc(byte_size);

    srand(time(NULL)); // Seed random number generator

    for (int i = 0; i < data_size; i++) host_input[i] = rand() % 10;

    dim3 block(block_size);
    dim3 grid((data_size / byte_size) / parallel_reduction);

    int temp_size = sizeof(int) * grid.x;
    host_ref = (int*)malloc(temp_size);

    int* device_input, * device_temp;

    cudaMalloc((void**)&device_input, byte_size);
    cudaMalloc((void**)&device_temp, temp_size);

    cudaMemset(device_temp, 0, temp_size);
    cudaMemcpy(device_input, host_input, byte_size, cudaMemcpyHostToDevice);

    if (parallel_reduction == 2)
        unrolling2_kernel << < grid, block >> > (device_input, device_temp, data_size);

    cudaDeviceSynchronize();
    cudaMemcpy(host_ref, device_temp, temp_size, cudaMemcpyDeviceToHost);

    int gpu_result = 0;
    for (int i = 0; i < grid.x; i++) {
        gpu_result += host_ref[i];
    }

    cudaFree(device_input);
    cudaFree(device_temp);
    free(host_input);
    free(host_ref);
    
    return 0;
}
