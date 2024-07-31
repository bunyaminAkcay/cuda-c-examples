#include "MatrixParser.h"
#include "MatrixHelper.h"
#include "cuSparseKernels.cuh"
#include "CsrBasedSpMM.cuh"

#include <string>

typedef float element_t;


struct testMatrix
{
    std::string fileName;
    bool symmetrical;
};


testMatrix matrices[] = { testMatrix{"1138_bus.mtx", true}, {"b1_ss.mtx", false}, {"bcsstk01.mtx", true}};


void testCsrBasedSpMM(const testMatrix &testMatrix){
        
        CooMatrixParser<element_t> cooMatrixParser(testMatrix.fileName, testMatrix.symmetrical, Order::rowMajor);
        //cooMatrixParser.saveSparseMatrixAsPPM3Image("matrixImages/b1_ss");
        
        size_t m = cooMatrixParser.getRowCount();
        size_t k = cooMatrixParser.getColCount();
        size_t n = m;
        size_t nnz = cooMatrixParser.getNNZ();

        element_t alpha = static_cast<element_t>(1);
        element_t beta = static_cast<element_t>(0);

        int* h_A_rowIds = cooMatrixParser.rowIds;
        int* h_A_colIds = cooMatrixParser.colIds;
        element_t* h_A_values = cooMatrixParser.values; 

        element_t *h_B = (element_t *)malloc(k * n * sizeof(element_t));
        element_t *h_C = (element_t *)malloc(m * n * sizeof(element_t));
        element_t *h_D_cuSparse = (element_t *)malloc(m * n * sizeof(element_t));
        element_t *h_D_csrBased = (element_t *)malloc(m * n * sizeof(element_t));
        
        MatrixHelper<element_t>::initRandomDenseMatrix(h_B, k, n);
        MatrixHelper<element_t>::initZeroMatrix(h_C, m, n);
        MatrixHelper<element_t>::initZeroMatrix(h_D_cuSparse, m, n);
        
        //perform cuSparse
        cuSparseSpMMwithCOO<element_t>(h_A_rowIds, h_A_colIds, h_A_values, h_B, h_C, h_D_cuSparse, m, k, n, nnz, alpha, beta);


        //perform Csr based SpMM
        MatrixHelper<element_t>::initZeroMatrix(h_C, m, n);
        MatrixHelper<element_t>::initZeroMatrix(h_D_csrBased, m, n);

        CsrBasedSpMM<element_t>(h_A_rowIds, h_A_colIds, h_A_values, h_B, h_C, h_D_csrBased, m, k, n, nnz, alpha, beta);

        free(h_B);
        free(h_C);
        

        size_t wrongElementCount;
        //compare results
        for (int rowId = 0; rowId < m; rowId++)
        {
            for (int colId = 0; colId < n; colId++)
            {
                if (h_D_cuSparse[rowId * n + colId] != h_D_csrBased[rowId * n + colId])
                {
                    wrongElementCount++;
                }
                
            }
        }
        
        printf("Correctness: %d/%d", (m * n - wrongElementCount)/(m * n));

        free(h_D_cuSparse);
        free(h_D_csrBased);
}

int main()
{

    for(const testMatrix &testMatrix : matrices){
        testCsrBasedSpMM(testMatrix);
    }
    
    return 0;
}

/*
    if(printResults && cooMatrixParser.sparseMatrixToClassicMatrix()){
        MatrixHelper<element_t>::printResult(m, n, k, cooMatrixParser.getClassicMatrix(), h_B, h_C, h_D_cuSparse);
    } 
*/