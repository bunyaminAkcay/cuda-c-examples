#include <cstdio>
#include <iostream>
#include <string>

#define N 256
#define blockSize 16

typedef float elem_t;

__global__
void vecAddKernel(elem_t* A_d, elem_t* B_d, elem_t* C_d, int len){

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if ( index < len)
    {
        C_d[index] = A_d[index] + B_d[index];
    }
    
}

void vecAdd(elem_t* A_h, elem_t* B_h, elem_t* C_h, int len){
    elem_t *A_d, *B_d, *C_d;
    int size = len * sizeof(elem_t);

    cudaMalloc((void **) &A_d, size);
    cudaMalloc((void **) &B_d, size);
    cudaMalloc((void **) &C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    vecAddKernel<<<(len + blockSize -1)/blockSize, blockSize>>>(A_d, B_d, C_d, len);

    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

void printResults(elem_t* A_h, elem_t* B_h, elem_t* C_h, int len){
    for (size_t i = 0; i < len; i++)
    {
        printf("%f + %f = %f\n", A_h[i], B_h[i], C_h[i]);
    }
}

void testResults(elem_t* C_h, int len){
    bool testPassed = true;
    

    for (size_t i = 0; i < N; i++)
    {
        if (C_h[i] != N)
        {
            testPassed = false;
            break;
        }       
    }

    std::string resultText = testPassed ? "Test passed" : "Test failed";
    std::cout << resultText << std::endl;
}

int main(){

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if ( deviceCount == 0){
        std::cout << "No cuda device found!" << std::endl;
        exit(-1);
    }

    cudaSetDevice(0);

    elem_t A_h[N], B_h[N], C_h[N];

    for (size_t i = 0; i < N; i++)
    {
        A_h[i] = i;
        B_h[i] = N - i;
    }

    vecAdd(A_h, B_h, C_h, N);

    printResults(A_h, B_h, C_h, N);
    testResults(C_h, N);

    return 0;
}