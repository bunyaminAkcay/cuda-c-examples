#include "MatrixReader.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath>
#include <stdio.h>
#include <cassert>
#include <cstring>

template<typename T>
void MatrixReader<T>::readMatrixFromFile(std::string fileName){
    
    std::ifstream infile(fileName);
    if (!infile.is_open()) {
        std::cerr << "Error opening file" << std::endl;
        exit(-1);
    }

    std::string line;
    
    while (std::getline(infile, line)) {
        if (line[0] != '%' && !line.empty()) {
            break;
        }
    }

    std::istringstream iss(line);
    
    iss >> rowCount >> colCount >> nnz;

    if (symmetrical)
    {
        // actually for symmetrical case, nnz is equals 2*nnz - nnz_on_diagonal 
        _rowIds = new size_t[2 * nnz];
        _colIds = new size_t[2 * nnz];
        _values = new T [2 * nnz];
    }
    else{
        _rowIds = new size_t[nnz];
        _colIds = new size_t[nnz];
        _values = new T [nnz];
    }
    


    bool hasValue = false;

    std::streampos pos = infile.tellg();
    std::getline(infile, line);
    std::istringstream test_iss(line);
    size_t testRowId, testColId;
    T testValue;

    int index = 0;

    if (test_iss >> testRowId >> testColId >> testValue) {
        hasValue = true;
        _rowIds[index] = testRowId - 1;
        _colIds[index] = testColId - 1;
        _values[index] = testValue;
        maxValue = abs(testValue);
        index++;
        
    } else {
        _rowIds[index] = testRowId - 1;
        _colIds[index] = testColId - 1;
        maxValue = 1;
        index++;
    }

    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);

        size_t rowId, colId;
        T value;
        iss >> rowId >> colId;
        _rowIds[index] = rowId - 1;
        _colIds[index] = colId - 1;
        if (hasValue) {
            iss >> value;
            if (abs(value) > maxValue)
            {
                maxValue = abs(value);
            }
            _values[index] = value;
        }
        else {
            _values[index] = static_cast<T>(1);
        }

        index++;

        if (symmetrical)
        {
            _rowIds[index] = colId - 1;
            _colIds[index] = rowId - 1;
            

            if (hasValue) {
                
                _values[index] = value;
            }
            else {
                _values[index] = static_cast<T>(1);
            }
            index++;   
        }
        
    }

    if (!symmetrical)
    {
        assert(nnz == index);
    }

    nnz = index;
    
    if (symmetrical)
    {
        size_t* new_rowIds = new size_t[nnz];
        size_t* new_colIds = new size_t[nnz];
        T* new_values = new T[nnz];
        memcpy(new_rowIds, _rowIds, nnz * sizeof(size_t));
        memcpy(new_colIds, _colIds, nnz * sizeof(size_t));
        memcpy(new_values, _values, nnz * sizeof(T));

        delete[] _rowIds;
        delete[] _colIds;
        delete[] _values;

        _rowIds = new_rowIds;
        _colIds = new_colIds;
        _values = new_values;
    }
    
    
    infile.close();
}

template<typename T>
size_t MatrixReader<T>::getRowCount(){
    return rowCount;
}

template<typename T>
size_t MatrixReader<T>::getColCount(){
    return colCount;
}

template<typename T>
size_t MatrixReader<T>::getNNZ(){
    return nnz;
}

template<typename T>
bool MatrixReader<T>::sparseMatrixToClassicMatrix(){

    if (rowCount > maxMatrixSizeToConvertNormalMatrix || colCount > maxMatrixSizeToConvertNormalMatrix)
    {
        printf("Matrix is too big to store as classic way. Max matrix size is %d", int(maxMatrixSizeToConvertNormalMatrix) );
        return false;
    }
    

    classicMatrix = new T[rowCount * colCount];
    classicMatrixAllocated = true;
    
    for (int nzi = 0; nzi < nnz; nzi++)
    {
        classicMatrix[_rowIds[nzi] * colCount + _colIds[nzi]] = _values[nzi];
    }

    return true;
}



template<typename T>
MatrixReader<T>::~MatrixReader(){

    free(_rowIds);
    free(_colIds);
    free(_values);
    free(classicMatrix);
}

template<class T>
CooMatrixReader<T>::CooMatrixReader(std::string fileName, bool symmetrical){
    this->symmetrical = symmetrical;
    this->readMatrixFromFile(fileName);
    rowIds = this->_rowIds;
    colIds = this->_colIds;
    values = this->_values;
}

template<typename T>
void CooMatrixReader<T>::plotMatrix(){

}

template<class T>
bool CooMatrixReader<T>::saveSparseMatrixAsPPM3Image(std::string fileName){

    if (!this->sparseMatrixToClassicMatrix())
    {
        printf("Matrix is too big to save as ppm.");
        return false;
    }
    
    std::ofstream ppm3ImageFile;
    ppm3ImageFile.open ( fileName + ".ppm");

    ppm3ImageFile << "P3\n" << this->rowCount << " " << this->colCount << "\n" << 255 << "\n";

    for (size_t j = 0; j < this->rowCount; j++)
    {
        for (size_t i = 0; i < this->colCount; i++)
        {
            T value = this->classicMatrix[j * this->colCount + i];
            if (value == static_cast<T>(0))
            {
                ppm3ImageFile << "255 255 255 ";
            }
            else
            {
                float normalizedValue = float(abs(value))/this->maxValue;
                float sigmoid = 1 / (1 + exp(-10*normalizedValue));
                sigmoid = 2 * (sigmoid - 0.5);
                uint r = 68 + sigmoid * 185;
                uint g = 1 + sigmoid * 230;
                uint b = 84 - sigmoid * 48;
                ppm3ImageFile << r << " " << g << " " << b << " ";
            }
        }
        ppm3ImageFile << "\n";
    }
    
    ppm3ImageFile.close();
    return true;
}

template<class T>
CooMatrixReader<T>::~CooMatrixReader(){}

template class MatrixReader<float>;
template class MatrixReader<double>;

template class CooMatrixReader<float>;
template class CooMatrixReader<double>;