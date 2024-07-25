#include <cstdio>

typedef float element_t;

template <typename T>
__host__ __device__ T blockCeil(T const n, T const blockSize){
    return (n + blockSize -1)/blockSize;
}

/*
    C = alpha * AB + beta * C

    dimension of A is m * k
    dimension of B is k * n
    dimension of C is m * n
*/


/*
    BLOCK_TILE_X, BLOCK_TILE_K, BLOCK_TILE_Y is given
    as templete parameter because these are compile-time
    constant. If we use blockDim.x, blockDim.y instead
    of that, we cannot unroll loops due to blockDim.x
    and blockDim.y is run-time constant.
*/
template <typename T, size_t BLOCK_TILE_Y, size_t BLOCK_TILE_K, size_t BLOCK_TILE_X>
__global__ void gemm2dBlockTilingKernel(size_t const m,
                                        size_t const n,
                                        size_t const k,
                                        T const alpha,
                                        T const beta,
                                        T const* A,
                                        T const* B,
                                        T* C)
{
    //cCol is the x position of thread
    size_t const cColIdx = blockIdx.x * blockDim.x + threadIdx.x;
    //cRow is the y position of thread
    size_t const cRowIdx = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ T aTile[BLOCK_TILE_Y][BLOCK_TILE_K];
    __shared__ T bTile[BLOCK_TILE_K][BLOCK_TILE_X];

    size_t const numberOfTiles = blockCeil(k, BLOCK_TILE_K);
    T value = static_cast<T>(0);
    for (size_t tileId = 0; tileId < numberOfTiles; tileId++)
    {
        //load data from dram to shared memory
        if (cRowIdx < m && (tileId * BLOCK_TILE_K + threadIdx.x) < k){
            aTile[threadIdx.y][threadIdx.x] = A[cRowIdx * k + tileId * BLOCK_TILE_K + threadIdx.x];
        }
        else{
            aTile[threadIdx.y][threadIdx.x] = 0;
        }

        if ((tileId * BLOCK_TILE_K + threadIdx.y) < k && cColIdx < n){
            bTile[threadIdx.y][threadIdx.x] = B[(tileId * BLOCK_TILE_K + threadIdx.y) * n + cColIdx];
        }
        else{
            bTile[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();
        
        //add to value
        #pragma unroll
        for (size_t ki = 0; ki < BLOCK_TILE_K; ki++)
        {
            value += aTile[threadIdx.y][ki] * bTile[ki][threadIdx.x];
        }
        __syncthreads();
    }

    if (cRowIdx < m && cColIdx < n){
        C[cRowIdx * n + cColIdx] = alpha * value + beta * C[cRowIdx * n + cColIdx];
    }

}

template<typename T>
void gemm2dBlockTiling( size_t m,
                size_t n,
                size_t k,
                T const alpha,
                T const beta,
                T const* h_A,
                T const* h_B,
                T const* h_C,
                T *h_D)
{
    T *d_A, *d_B, *d_C;
    size_t const blockWidth = 32;

    cudaMalloc((void**) &d_A, m * k * sizeof(T));
    cudaMalloc((void**) &d_B, k * n * sizeof(T));
    cudaMalloc((void**) &d_C, m * n * sizeof(T));

    cudaMemcpy(d_A, h_A, m * k * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, m * n * sizeof(T), cudaMemcpyHostToDevice);

    dim3 const dimBlock(blockWidth, blockWidth, 1U);
    dim3 const dimGrid(blockCeil(static_cast<uint>(n), dimBlock.x), blockCeil(static_cast<uint>(m), dimBlock.y), 1U);

    gemm2dBlockTilingKernel<T, blockWidth, blockWidth, blockWidth><<<dimGrid, dimBlock>>>(m, n, k, alpha, beta, d_A, d_B, d_C);

    cudaMemcpy(h_D, d_C, m * n * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

template<typename T>
void initRandomMatrix(T *matrix, size_t const rowCount, size_t const colCount)
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

int main(){

    bool const printResults = false;

    size_t m = 1024, k = 1024, n = 1024;
    element_t alpha = 1, beta = 0;

    element_t *h_A = (element_t *)malloc(m * k * sizeof(element_t));
    element_t *h_B = (element_t *)malloc(k * n * sizeof(element_t));
    element_t *h_C = (element_t *)malloc(m * n * sizeof(element_t));
    element_t *h_D = (element_t *)malloc(m * n * sizeof(element_t));

    initRandomMatrix<element_t>(h_A, m, k);
    initRandomMatrix<element_t>(h_B, k, n);
    initZeroMatrix<element_t>(h_C, m, n);
    initZeroMatrix<element_t>(h_D, m, n);
    
    gemm2dBlockTiling<element_t>(m, n, k, alpha, beta, h_A, h_B, h_C, h_D);

    if (printResults)
    {
        printResult<element_t>(m, n, k, h_A, h_B, h_C, h_D);
    }
    
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);

    return 0;
}