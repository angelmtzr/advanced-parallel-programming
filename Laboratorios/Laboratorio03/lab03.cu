#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define MATRIX_SIZE 2
#define BLOCK_SIZE 16

// Kernel for matrix addition in CUDA
__global__ void matrixAddKernel(int* a, int* b, int* c, int size) {
    int threadX = threadIdx.x;
    int threadY = threadIdx.y;

    int blockX = blockIdx.x;
    int blockY = blockIdx.y;

    int dimX = blockDim.x;
    int dimY = blockDim.y;

    int globalIdx = blockX * dimX + threadX;
    int globalIdy = blockY * dimY + threadY;

    int globalId = (globalIdy * blockDim.x * gridDim.x) + globalIdx;

    c[globalId] = a[globalId] + b[globalId];
}

// Function to initialize a matrix with random values
void initializeMatrix(int* matrix, int col, int row, int number) {
    for (long int i = 0; i < row * col; i++) {
            matrix[i] = number;
    }
}

void printMatrix(int* matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("Matrix Value: %d \n", matrix[i * size + j]);
        }
    }
}

int main() {
    // Host matrices variables
    int* hostMatrixA, * hostMatrixB, * hostMatrixC;
    int* hostMatrixA2, * hostMatrixB2, * hostMatrixC2;

    // Device matrices variables
    int* deviceMatrixA, * deviceMatrixB, * deviceMatrixC;
    int* deviceMatrixA2, * deviceMatrixB2, * deviceMatrixC2;
    
    // Matrix size
    int col = 100;
    int row = 100;

    int col2 = 2;
    int row2 = 2;
    
    // Size in bytes
    long int bytes = col * row * sizeof(int);
    long int bytes2 = col2 * row2 * sizeof(int);

    // Memory allocation on the host
    hostMatrixA = (int*)malloc(bytes);
    hostMatrixB = (int*)malloc(bytes);
    hostMatrixC = (int*)malloc(bytes);

    hostMatrixA2 = (int*)malloc(bytes2);
    hostMatrixB2 = (int*)malloc(bytes2);
    hostMatrixC2 = (int*)malloc(bytes2);

    // Initialize matrices hostMatrixA and hostMatrixB with random values
    initializeMatrix(hostMatrixA, row, col, 1);
    initializeMatrix(hostMatrixB, row, col, 2);

    // Initialize matrices hostMatrixA2 and hostMatrixB2 with random values
    initializeMatrix(hostMatrixA2, row2, col2, 1);
    initializeMatrix(hostMatrixB2, row2, col2, 2);

    // Allocate memory on the device
    cudaMalloc(&deviceMatrixA, bytes);
    cudaMalloc(&deviceMatrixB, bytes);
    cudaMalloc(&deviceMatrixC, bytes);

    cudaMalloc(&deviceMatrixA2, bytes2);
    cudaMalloc(&deviceMatrixB2, bytes2);
    cudaMalloc(&deviceMatrixC2, bytes2);

    // Copy data from host to device
    cudaMemcpy(deviceMatrixA, hostMatrixA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrixB, hostMatrixB, bytes, cudaMemcpyHostToDevice);

    cudaMemcpy(deviceMatrixA2, hostMatrixA2, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrixB2, hostMatrixB2, bytes, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockSize(1, 1, 1);
    dim3 gridSize(1, 1, 1);

    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();

    // Launch the matrix addition kernel
    matrixAddKernel << <gridSize, blockSize >> > (deviceMatrixA, deviceMatrixB, deviceMatrixC, row);
    //matrixMultiplyKernel << <gridSize, blockSize >> > (deviceMatrixA2, deviceMatrixB2, deviceMatrixC2, 2);
    cudaDeviceSynchronize();

    // Measure execution time
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
    }

    // Copy results from device to host
    cudaMemcpy(hostMatrixC, deviceMatrixC, bytes, cudaMemcpyDeviceToHost);

    cudaMemcpy(hostMatrixC2, deviceMatrixC2, bytes, cudaMemcpyDeviceToHost);

    // Check that the matrices are computed correctly
    //printMatrix(hostMatrixC, row);

    //printMatrix(hostMatrixC2, row2);

    // Free memory
    free(hostMatrixA);
    free(hostMatrixB);
    free(hostMatrixC);
    cudaFree(deviceMatrixA);
    cudaFree(deviceMatrixB);
    cudaFree(deviceMatrixC);

    free(hostMatrixA2);
    free(hostMatrixB2);
    free(hostMatrixC2);
    cudaFree(deviceMatrixA2);
    cudaFree(deviceMatrixB2);
    cudaFree(deviceMatrixC2);

    return 0;
}
