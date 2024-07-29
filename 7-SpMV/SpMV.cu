#include "MatrixReader.h"

typedef float element_t;

int main()
{
    CooMatrixReader<element_t> cooMatrixReader("bcsstk01.mtx", false);
    cooMatrixReader.saveSparseMatrixAsPPM3Image("bcsstk01");
    return 0;
}