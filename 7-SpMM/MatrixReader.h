#include <string>

enum Order{rowMajor, colMajor};

template<typename T>
class MatrixReader{

    public:
        virtual void plotMatrix() = 0;
        virtual bool saveSparseMatrixAsPPM3Image(std::string fileName) = 0;
        size_t getRowCount();
        size_t getColCount();
        size_t getNNZ();
        ~MatrixReader();
        bool sparseMatrixToClassicMatrix();//return true if success
        T* getClassicMatrix();
        void printCoordinates();
        void sort(Order order);

    protected:
        bool symmetrical;
        int *_rowIds, *_colIds;
        int rowCount, colCount, nnz;
        T maxValue = 0;
        T *_values;
        T *classicMatrix;
        void readMatrixFromFile(std::string fileName);
        
    private:
        bool classicMatrixAllocated = false;
        size_t maxMatrixSizeToConvertNormalMatrix = 2048;
};

template<typename T>
class CooMatrixReader : public MatrixReader<T>{

    public:
        int *rowIds, *colIds;
        T *values;
        CooMatrixReader(std::string fileName, bool symmetrical, Order order);
        ~CooMatrixReader();
        void plotMatrix();
        bool saveSparseMatrixAsPPM3Image(std::string fileName);
};