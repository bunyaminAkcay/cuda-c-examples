#include "MatrixReader.h"

typedef float element_t;

int main()
{
    CooMatrixReader<element_t> cooMatrixReader("1138_bus.mtx", true);
    cooMatrixReader.saveSparseMatrixAsPPM3Image("1138_bus");
    
    return 0;
}