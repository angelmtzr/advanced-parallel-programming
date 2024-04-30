#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>

void printMatrix(int* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%i, ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void convolveKernel(int* inputMatrix, int* outputMatrix, int* filter, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int id = (row * cols) + col;
    int total = 0;

    if (row == 0) {
        if (col == 0) {
            total += inputMatrix[row * cols + col] * filter[4];
            total += inputMatrix[row * cols + col + 1] * filter[5];
            total += inputMatrix[(row + 1) * cols + col] * filter[7];
            total += inputMatrix[(row + 1) * cols + col + 1] * filter[8];
            outputMatrix[id] = total;
        }
        else if (col == cols - 1) {
            total += inputMatrix[row * cols + col - 1] * filter[3];
            total += inputMatrix[row * cols + col] * filter[4];
            total += inputMatrix[(row + 1) * cols + col - 1] * filter[6];
            total += inputMatrix[(row + 1) * cols + col] * filter[7];
            outputMatrix[id] = total;
        }
        else {
            total += inputMatrix[row * cols + col - 1] * filter[3];
            total += inputMatrix[row * cols + col] * filter[4];
            total += inputMatrix[row * cols + col + 1] * filter[5];
            total += inputMatrix[(row + 1) * cols + col - 1] * filter[6];
            total += inputMatrix[(row + 1) * cols + col] * filter[7];
            total += inputMatrix[(row + 1) * cols + col + 1] * filter[8];
            outputMatrix[id] = total;
        }
    }
    else if (row == rows - 1) {
        if (col == 0) {
            total += inputMatrix[(row - 1) * cols + col] * filter[1];
            total += inputMatrix[(row - 1) * cols + col + 1] * filter[2];
            total += inputMatrix[row * cols + col] * filter[4];
            total += inputMatrix[row * cols + col + 1] * filter[5];
            outputMatrix[id] = total;
        }
        else if (col == cols - 1) {
            total += inputMatrix[(row - 1) * cols + col - 1] * filter[0];
            total += inputMatrix[(row - 1) * cols + col] * filter[1];
            total += inputMatrix[row * cols + col - 1] * filter[3];
            total += inputMatrix[row * cols + col] * filter[4];
            outputMatrix[id] = total; 
        }
        else {
            total += inputMatrix[(row - 1) * cols + col - 1] * filter[0];
            total += inputMatrix[(row - 1) * cols + col] * filter[1];
            total += inputMatrix[(row - 1) * cols + col + 1] * filter[2];
            total += inputMatrix[row * cols + col - 1] * filter[3];
            total += inputMatrix[row * cols + col] * filter[4];
            total += inputMatrix[row * cols + col + 1] * filter[5];
            outputMatrix[id] = total;
        }
    }
    else {
        if (col == 0) {
            total += inputMatrix[(row - 1) * cols + col] * filter[1];
            total += inputMatrix[(row - 1) * cols + col + 1] * filter[2];
            total += inputMatrix[row * cols + col] * filter[4];
            total += inputMatrix[row * cols + col + 1] * filter[5];
            total += inputMatrix[(row + 1) * cols + col] * filter[7];
            total += inputMatrix[(row + 1) * cols + col + 1] * filter[8];
            outputMatrix[id] = total;
        }
        else if (col == cols - 1) {
            total += inputMatrix[(row - 1) * cols + col - 1] * filter[0];
            total += inputMatrix[(row - 1) * cols + col] * filter[1];
            total += inputMatrix[row * cols + col - 1] * filter[3];
            total += inputMatrix[row * cols + col] * filter[4];
            total += inputMatrix[(row + 1) * cols + col - 1] * filter[6];
            total += inputMatrix[(row + 1) * cols + col] * filter[7];
            outputMatrix[id] = total;
        }
        else {
            total += inputMatrix[(row - 1) * cols + col - 1] * filter[0];
            total += inputMatrix[(row - 1) * cols + col] * filter[1];
            total += inputMatrix[(row - 1) * cols + col + 1] * filter[2];
            total += inputMatrix[row * cols + col - 1] * filter[3];
            total += inputMatrix[row * cols + col] * filter[4];
            total += inputMatrix[row * cols + col + 1] * filter[5];
            total += inputMatrix[(row + 1) * cols + col - 1] * filter[6];
            total += inputMatrix[(row + 1) * cols + col] * filter[7];
            total += inputMatrix[(row + 1) * cols + col + 1] * filter[8];
            outputMatrix[id] = total;
        }
    }
}

void cpuConvolution(int* inputMatrix, int* outputMatrix, int* filter, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int total = 0;
            int id = (i * cols) + j;

            if (i == 0) {
                if (j == 0) {
                    total += inputMatrix[i * cols + j] * filter[4];
                    total += inputMatrix[i * cols + j + 1] * filter[5];
                    total += inputMatrix[(i + 1) * cols + j] * filter[7];
                    total += inputMatrix[(i + 1) * cols + j + 1] * filter[8];
                    outputMatrix[id] = total;
                }
                else if (j == cols - 1) {
                    total += inputMatrix[i * cols + j - 1] * filter[3];
                    total += inputMatrix[i * cols + j] * filter[4];
                    total += inputMatrix[(i + 1) * cols + j - 1] * filter[6];
                    total += inputMatrix[(i + 1) * cols + j] * filter[7];
                    outputMatrix[id] = total;
                }
                else {
                    total += inputMatrix[i * cols + j - 1] * filter[3];
                    total += inputMatrix[i * cols + j] * filter[4];
                    total += inputMatrix[i * cols + j + 1] * filter[5];
                    total += inputMatrix[(i + 1) * cols + j - 1] * filter[6];
                    total += inputMatrix[(i + 1) * cols + j] * filter[7];
                    total += inputMatrix[(i + 1) * cols + j + 1] * filter[8];
                    outputMatrix[id] = total;
                }
            }
            else if (i == rows - 1) {
                if (j == 0) {
                    total += inputMatrix[(i - 1) * cols + j] * filter[1];
                    total += inputMatrix[(i - 1) * cols + j + 1] * filter[2];
                    total += inputMatrix[i * cols + j] * filter[4];
                    total += inputMatrix[i * cols + j + 1] * filter[5];
                    outputMatrix[id] = total;
                }
                else if (j == cols - 1) {
                    total += inputMatrix[(i - 1) * cols + j - 1] * filter[0];
                    total += inputMatrix[(i - 1) * cols + j] * filter[1];
                    total += inputMatrix[i * cols + j - 1] * filter[3];
                    total += inputMatrix[i * cols + j] * filter[4];
                    outputMatrix[id] = total; 
                }
                else {
                    total += inputMatrix[(i - 1) * cols + j - 1] * filter[0];
                    total += inputMatrix[(i - 1) * cols + j] * filter[1];
                    total += inputMatrix[(i - 1) * cols + j + 1] * filter[2];
                    total += inputMatrix[i * cols + j - 1] * filter[3];
                    total += inputMatrix[i * cols + j] * filter[4];
                    total += inputMatrix[i * cols + j + 1] * filter[5];
                    outputMatrix[id] = total;
                }
            }
            else {
                if (j == 0) {
                    total += inputMatrix[(i - 1) * cols + j] * filter[1];
                    total += inputMatrix[(i - 1) * cols + j + 1] * filter[2];
                    total += inputMatrix[i * cols + j] * filter[4];
                    total += inputMatrix[i * cols + j + 1] * filter[5];
                    total += inputMatrix[(i + 1) * cols + j] * filter[7];
                    total += inputMatrix[(i + 1) * cols + j + 1] * filter[8];
                    outputMatrix[id] = total;
                }
                else if (j == cols - 1) {
                    total += inputMatrix[(i - 1) * cols + j - 1] * filter[0];
                    total += inputMatrix[(i - 1) * cols + j] * filter[1];
                    total += inputMatrix[i * cols + j - 1] * filter[3];
                    total += inputMatrix[i * cols + j] * filter[4];
                    total += inputMatrix[(i + 1) * cols + j - 1] * filter[6];
                    total += inputMatrix[(i + 1) * cols + j] * filter[7];
                    outputMatrix[id] = total;
                }
                else {
                    total += inputMatrix[(i - 1) * cols + j - 1] * filter[0];
                    total += inputMatrix[(i - 1) * cols + j] * filter[1];
                    total += inputMatrix[(i - 1) * cols + j + 1] * filter[2];
                    total += inputMatrix[i * cols + j - 1] * filter[3];
                    total += inputMatrix[i * cols + j] * filter[4];
                    total += inputMatrix[i * cols + j + 1] * filter[5];
                    total += inputMatrix[(i + 1) * cols + j - 1] * filter[6];
                    total += inputMatrix[(i + 1) * cols + j] * filter[7];
                    total += inputMatrix[(i + 1) * cols + j + 1] * filter[8];
                    outputMatrix[id] = total;
                }
            }
        }
    }
}

void fillMatrix(int* matrix, int rows, int cols, int num) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * rows + j] = num;
        }
    }
}

int main()
{
    clock_t cpuStart, cpuStop, gpuStart, gpuStop;
    int* CmatrixA, *GmatrixA;
    int* CresultA, *GresultA;
    int N = 4, M = 5;
    int* filterDevice;

    int filter[3][3] = { {0,  1, 0}, 
                         {1, -4, 1}, 
                         {0,  1, 0} };
    int matrixT[4][5] = {{1, 0, 1, 2, 2},
                         {1, 1, 2, 2, 3},
                         {1, 2, 2, 6, 3}, 
                         {1, 1, 2, 2, 3}};
    
    int matrixZ[4][5] = {{0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0}};

    int size = N * M * sizeof(int);

    CmatrixA = (int*)malloc(size);
    CresultA = (int*)malloc(size);

    CmatrixA = &matrixT[0][0];
    filterDevice = &filter[0][0];

    printMatrix(&matrixT[0][0], N, M);
    cpuStart = clock();
    cpuConvolution(&matrixT[0][0], &matrixZ[0][0], &filter[0][0], N, M);
    cpuStop = clock();
    printMatrix(&matrixZ[0][0], N, M);

    cudaMalloc((void**)&GmatrixA, size);
    cudaMalloc((void**)&GresultA, size);
    cudaMalloc((void**)&filterDevice, size);

    cudaMemcpy(GmatrixA, CmatrixA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(GresultA, CresultA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(filterDevice, filter, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    gpuStart = clock();
    convolveKernel << <numBlocks, threadsPerBlock >> > (GmatrixA, GresultA, filterDevice, N, M);
    gpuStop = clock();

    cudaMemcpy(CmatrixA, GmatrixA, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(CresultA, GresultA, size, cudaMemcpyDeviceToHost);

    printMatrix(CresultA, N, M);

    cudaDeviceReset();
    cudaFree(GmatrixA);
    cudaFree(GresultA);

    cudaDeviceSynchronize();

    printf("CPU Time: %d \n", (cpuStop - cpuStart));
    printf("GPU Time: %d \n", (gpuStop - gpuStart));
    printf("Current Time: %d", clock());

    return 0;
}
