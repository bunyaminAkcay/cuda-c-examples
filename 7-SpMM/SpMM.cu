#include "MatrixReader.h"
#include <cuda_runtime.h>
#include <cusparse.h>

typedef float element_t;

void cuSparseSpMMwithCOO(int* h_A_rowIds, int* h_A_colIds, element_t* h_A_values, element_t* h_B, element_t* h_C, element_t* h_D, size_t m, size_t k, size_t n, size_t nnz, element_t alpha, element_t beta){
    
    element_t *d_B, *d_C;
    element_t *d_A_colIds, *d_A_rowIds, *d_A_values; 

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    cudaMalloc((void**) &d_B, k * n * sizeof(element_t));
    cudaMalloc((void**) &d_C, m * n * sizeof(element_t));
    cudaMalloc((void**) &d_A_colIds, nnz * sizeof(int));
    cudaMalloc((void**) &d_A_rowIds, nnz * sizeof(int));
    cudaMalloc((void**) &d_A_values, nnz * sizeof(int));

    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "CUDA memory allocation failed\n");
        return;
    }

    cudaMemcpy(d_B, h_B, k * n * sizeof(element_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, m * n * sizeof(element_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_colIds, h_A_colIds, nnz * sizeof(element_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_rowIds, h_A_rowIds, nnz * sizeof(element_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_values, h_A_values, nnz * sizeof(element_t), cudaMemcpyHostToDevice);

    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "CUDA memory copy failed\n");
        return;
    }

    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;

    cusparseCreateCoo(&matA, m, k, nnz, d_A_rowIds, d_A_colIds, d_A_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseCreateDnMat(&matB, k, n, n, d_B, CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&matC, m, n, n, d_C, CUDA_R_32F, CUSPARSE_ORDER_ROW);

    size_t bufferSize;
    void *dBuffer;
    cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    // Perform SpMM
    cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);


    cudaMemcpy(h_D, d_C, m * n * sizeof(element_t), cudaMemcpyDeviceToHost);

    // Free workspace
    

    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroy(handle);

    cudaFree(dBuffer);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_colIds);
    cudaFree(d_A_rowIds);
    cudaFree(d_A_values);

}

template<typename T>
void initRandomDenseMatrix(T *matrix, size_t const rowCount, size_t const colCount)
{
    for (size_t row = 0; row < rowCount; row++)
    {
        for (size_t col = 0; col < colCount; col++)
        {
            matrix[row * colCount + col] = rand() / static_cast<T>(RAND_MAX);
        }
    }
}

template<typename T>
void initZeroMatrix(T *matrix, size_t const rowCount, size_t const colCount)
{
    for (size_t row = 0; row < rowCount; row++)
    {
        for (size_t col = 0; col < colCount; col++)
        {
            matrix[row * colCount + col] = 0;
        }
    }
}

template<typename T>
void printResult(size_t const m, size_t const n, size_t const k, T const* A, T const* B, T const* C, T const* D)
{
    size_t const matrixCount = 4;
    T const* matrixReferences[matrixCount] = {A, B, C, D};
    
    char matrixNames[matrixCount] = {'A', 'B', 'C', 'D'};
    size_t matrixDimensions[matrixCount][2] = { {m, k}, {k, n}, {m, n}, {m, n}};

    for (size_t k = 0; k < matrixCount; k++)
    {
        printf("%c [\n\n\t", matrixNames[k]);
        for (size_t j = 0; j < matrixDimensions[k][0]; j++)
        {
            for (size_t i = 0; i < matrixDimensions[k][1]; i++)
            {
                printf("%f ", matrixReferences[k][matrixDimensions[k][1] * j + i]);
            }
            printf("\n\t");
        }
        printf("\n]\n");
    }
}

int main()
{
    bool symmetrical = false;
    bool const printResults = true;
    
    CooMatrixReader<element_t> cooMatrixReader("matrices/b1_ss.mtx", symmetrical, Order::rowMajor);
    
    cooMatrixReader.saveSparseMatrixAsPPM3Image("b1_ss");
    
    size_t m = cooMatrixReader.getRowCount();
    size_t k = cooMatrixReader.getColCount();
    size_t n = m;
    size_t nnz = cooMatrixReader.getNNZ();

    element_t alpha = static_cast<element_t>(1);
    element_t beta = static_cast<element_t>(0);

    int* h_A_rowIds = cooMatrixReader.rowIds;
    int* h_A_colIds = cooMatrixReader.colIds;
    element_t* h_A_values = cooMatrixReader.values; 

    element_t *h_B = (element_t *)malloc(k * n * sizeof(element_t));
    element_t *h_C = (element_t *)malloc(m * n * sizeof(element_t));
    element_t *h_D = (element_t *)malloc(m * n * sizeof(element_t));
    
    initRandomDenseMatrix(h_B, k, n);
    initZeroMatrix(h_C, m, n);
    initZeroMatrix(h_D, m, n);
    
    cuSparseSpMMwithCOO(h_A_rowIds, h_A_colIds, h_A_values, h_B, h_C, h_D, m, k, n, nnz, alpha, beta);

    if(printResults && cooMatrixReader.sparseMatrixToClassicMatrix()){
        printResult(m, n, k, cooMatrixReader.getClassicMatrix(), h_B, h_C, h_D);
    } 
    
    free(h_B);
    free(h_C);
    free(h_D);

    return 0;
}