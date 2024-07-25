#include <cstdio>
#include <iostream>

typedef float elem_t;

const int blockSize = 16;

void initRandomMatrix(elem_t *matrix, int n)
{
    for (size_t j = 0; j < n; j++)
    {
        for (size_t i = 0; i < n; i++)
        {
            matrix[n * j + i] = rand() / (elem_t)RAND_MAX;
        }
    }
}

void initZeroMatrix(elem_t *matrix, int n)
{
    for (size_t j = 0; j < n; j++)
    {
        for (size_t i = 0; i < n; i++)
        {
            matrix[n * j + i] = 0;
        }
    }
}

void printResult(elem_t *matrixA, elem_t *matrixB, elem_t *matrixC, int len)
{

    elem_t *matrixReferences[3] = {matrixA, matrixB, matrixC};
    char matrixNames[3] = {'A', 'B', 'C'};
    for (size_t k = 0; k < 3; k++)
    {
        printf("%c [\n\n\t", matrixNames[k]);
        for (size_t j = 0; j < len; j++)
        {
            for (size_t i = 0; i < len; i++)
            {
                printf("%f ", matrixReferences[k][len * j + i]);
            }
            printf("\n\t");
        }
        printf("\n]\n");
    }
}

__global__ void matMulKernel(elem_t *matrixA_h, elem_t *matrixB_h, elem_t *matrixC_h, int n, int *globalMemCallCount)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < n && row < n)
    {
        elem_t value = 0;
        for (size_t i = 0; i < n; i++)
        {
            value += matrixA_h[row * n + i] * matrixB_h[i * n + col];
            atomicAdd(globalMemCallCount, 2);
        }
        matrixC_h[row * n + col] = value;
    }
}

__global__ void matMulWithTilingKernel(elem_t *matrixA_d, elem_t *matrixB_d, elem_t *matrixC_d, int n, int *globalMemCallCount)
{

    __shared__ elem_t Atile[blockSize][blockSize];
    __shared__ elem_t Btile[blockSize][blockSize];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    elem_t value = 0;
    int numTiles = (n + blockSize -1) / blockSize;
    for (size_t j = 0; j < numTiles; j++)
    {
        if (row < n && (j * blockSize + tx) < n)
        {
            Atile[ty][tx] = matrixA_d[row * n + j * blockSize + tx];
            atomicAdd(globalMemCallCount, 1);
        }
        else
        {
            Atile[ty][tx] = 0;
        }

        if ((j * blockSize + ty) < n && col < n)
        {
            Btile[ty][tx] = matrixB_d[(j * blockSize + ty) * n + col];
            atomicAdd(globalMemCallCount, 1);
        }
        else
        {
            Btile[ty][tx] = 0;
        }
        
        __syncthreads();

        for (size_t k = 0; k < blockSize; k++)
        {
            value += Atile[ty][k] * Btile[k][tx];
        }
        __syncthreads();
    }
    if ((row < n) && (col < n)){
        matrixC_d[row * n + col] = value;
    }
}

int matMul(elem_t *matrixA_h, elem_t *matrixB_h, elem_t *matrixC_h, int n, bool withTiling)
{

    elem_t *matrixA_d, *matrixB_d, *matrixC_d;
    int *globalMemoryCallCount_d;
    int a = 0;
    int *globalMemoryCallCount_h = &a;
    size_t size = n * n * sizeof(elem_t);

    cudaMalloc((void **)&matrixA_d, size);
    cudaMalloc((void **)&matrixB_d, size);
    cudaMalloc((void **)&matrixC_d, size);
    cudaMalloc((void **)&globalMemoryCallCount_d, sizeof(int));

    cudaMemcpy(matrixA_d, matrixA_h, n * n * sizeof(elem_t), cudaMemcpyHostToDevice);
    cudaMemcpy(matrixB_d, matrixB_h, n * n * sizeof(elem_t), cudaMemcpyHostToDevice);

    cudaMemcpy(globalMemoryCallCount_d, 0, sizeof(elem_t), cudaMemcpyHostToDevice);

    dim3 dimBlock(blockSize, blockSize, 1);
    dim3 dimGrid((n + blockSize - 1) / blockSize, (n + blockSize - 1) / blockSize, 1);

    if (withTiling)
    {
        matMulWithTilingKernel<<<dimGrid, dimBlock>>>(matrixA_d, matrixB_d, matrixC_d, n, globalMemoryCallCount_d);
    }
    else
    {
        matMulKernel<<<dimGrid, dimBlock>>>(matrixA_d, matrixB_d, matrixC_d, n, globalMemoryCallCount_d);
    }

    cudaMemcpy(matrixC_h, matrixC_d, n * n * sizeof(elem_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(globalMemoryCallCount_h, globalMemoryCallCount_d, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(matrixA_d);
    cudaFree(matrixB_d);
    cudaFree(matrixC_d);
    cudaFree(globalMemoryCallCount_d);

    return *globalMemoryCallCount_h;
}

int main()
{
    int n = 512;

    elem_t *matrixA = (elem_t *)malloc(n * n * sizeof(elem_t));
    elem_t *matrixB = (elem_t *)malloc(n * n * sizeof(elem_t));
    elem_t *matrixC = (elem_t *)malloc(n * n * sizeof(elem_t));

    initRandomMatrix(matrixA, n);
    initRandomMatrix(matrixB, n);
    initZeroMatrix(matrixC, n);

    int matMulGlobalCallCount = matMul(matrixA, matrixB, matrixC, n, false);
    printf("MatMul:\n");
    //printResult(matrixA, matrixB, matrixC, n);

    initZeroMatrix(matrixC, n);

    printf("MatMul with tiling:\n");
    int matMulGlobalWithTilingCallCount = matMul(matrixA, matrixB, matrixC, n, true);
    //printResult(matrixA, matrixB, matrixC, n);

    printf("global read count of matmul without tiling:\t%d\n", matMulGlobalCallCount);
    printf("global read count of matmul with tiling:\t%d\n", matMulGlobalWithTilingCallCount);

    free(matrixA);
    free(matrixB);
    free(matrixC);

    return 0;
}