#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>


__global__ void transposeMatrixGPU(int* matrixA, int* matrixB, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows  && col < cols) {
        int idOriginal = (row * cols) + col;
        int idTransposed = (col * rows) + row;

        matrixB[idTransposed] = matrixA[idOriginal];
    }
}

void printMatrix(int* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%i, ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void transposeMatrixCPU(int* inputMatrix, int* outputMatrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            outputMatrix[rows * j + i] = inputMatrix[cols * i + j];
        }
    }
}

int main()
{

    //Data 
    int tempInput[] = { 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22};

    int numRows = 3, numCols = 4;
    int* inputMatrix, *outputMatrix;

    int size = numRows * numCols * sizeof(int);

    inputMatrix = (int*)malloc(size);
    outputMatrix = (int*)malloc(size);

    inputMatrix = &tempInput[0];

    //CPU
    printf("\n ORIGINAL: \n");
    printMatrix(inputMatrix, numRows, numCols);
    transposeMatrixCPU(inputMatrix, outputMatrix, numRows, numCols);
    printf("\n CPU: \n");
    printMatrix(outputMatrix, numCols, numRows);

    //GPU
    dim3 blockSize(4, 4, 1);
    dim3 gridSize(1, 1, 1);

    int* inputCPU;
    int* outputCPU;


    int* inputGPU;
    int* outputGPU;

    inputCPU = (int*)malloc(size);
    outputCPU = (int*)malloc(size);

    inputCPU = &tempInput[0];

    cudaMalloc((void**)&inputGPU, size);
    cudaMalloc((void**)&outputGPU, size);

    //transfer to GPU memory
    cudaMemcpy(inputGPU, inputCPU, size, cudaMemcpyHostToDevice);
    cudaMemcpy(outputGPU, outputCPU, size, cudaMemcpyHostToDevice);

    //kernel launch
    transposeMatrixGPU << <gridSize, blockSize >> > (inputGPU, outputGPU, numRows, numCols);


    //transfer to CPU host memory from GPU device //source, from, size
    cudaMemcpy(inputCPU, inputGPU, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(outputCPU, outputGPU, size, cudaMemcpyDeviceToHost);

    printf("\n GPU: \n");
    printMatrix(outputCPU, numCols, numRows);

    //cleanup
    cudaDeviceReset();
    cudaFree(inputGPU);
    cudaFree(outputGPU);



    cudaDeviceSynchronize();
    return 0;
}
