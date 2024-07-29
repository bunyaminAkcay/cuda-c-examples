#include <string>

template<typename T>
class MatrixReader{

    public:
        virtual void plotMatrix() = 0;
        virtual bool saveSparseMatrixAsPPM3Image(std::string fileName) = 0;
        size_t getRowCount();
        size_t getColCount();
        size_t getNNZ();
        ~MatrixReader();
    
    protected:
        bool symmetrical;
        size_t *_rowIds, *_colIds;
        size_t rowCount, colCount, nnz;
        T maxValue = 0;
        T *_values;
        T *classicMatrix;
        void readMatrixFromFile(std::string fileName);
        bool sparseMatrixToClassicMatrix();//return true if success
    private:
        bool classicMatrixAllocated = false;
        size_t maxMatrixSizeToConvertNormalMatrix = 2048;
};

template<typename T>
class CooMatrixReader : public MatrixReader<T>{

    public:
        size_t *rowIds, *colIds;
        T *values;
        CooMatrixReader(std::string fileName, bool symmetrical);
        ~CooMatrixReader();
        void plotMatrix();
        bool saveSparseMatrixAsPPM3Image(std::string fileName);
};