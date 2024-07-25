#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <fstream>

int blockSizeW = 16;

int imageWidth = 0, imageHeight = 0;

int* getPPMImageFromFile(std::string fileName){

    std::ifstream file(fileName);
    if (!file) {
        std::cerr << "Error opening file.\n";
        exit(-1);
    }
    
    std::string format;
    
    int width, height, max_color;
    
    file >> format >> width >> height >> max_color;
    
    if (format != "P3") {
        std::cerr << "Unsupported PPM format.\n";
        file.close();
        exit(-1);
    }
    
    int size = width * height * 3 * sizeof(int);
    int* image = (int *) malloc(size);


    for (int i = 0; i < width * height; ++i) {
        std::string r, g, b;
        file >> r >> g >> b;
        
        image[3*i    ] = std::stoi(r);
        image[3*i + 1] = std::stoi(g);
        image[3*i + 2] = std::stoi(b);
    }
    
    file.close();
    imageWidth = width;
    imageHeight = height;

    return image;
}

__global__
void getGrayscaleImageKernel(int* image_d, int* grayscaleImage_d, int imageW, int imageH){
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < imageW && row < imageH)
    {
        int pixelOffset = row * imageW + col;
        int r = image_d[pixelOffset * 3    ];
        int g = image_d[pixelOffset * 3 + 1];
        int b = image_d[pixelOffset * 3 + 2];
        grayscaleImage_d[pixelOffset] = 0.21f * r + 0.71 * g + 0.07*b;
    }
    
}

void getGrayscaleImage(int* image_h, int* grayscaleImage_h){
    int *image_d, *grayscaleImage_d;
    
    int giSize = imageHeight * imageWidth * sizeof(int);
    int iSize = 3 * giSize;

    cudaMalloc((void **) &image_d, iSize);
    cudaMalloc((void **) &grayscaleImage_d, giSize);

    cudaMemcpy(image_d, image_h, iSize, cudaMemcpyHostToDevice);

    dim3 dimBlock( blockSizeW, blockSizeW, 1);
    dim3 dimGrid( (imageWidth + blockSizeW -1 )/blockSizeW, (imageHeight + blockSizeW -1 )/blockSizeW, 1 );

    std::cout << "dimGrid:  " << dimGrid.x << " " << dimGrid.y << " " << dimGrid.z << std::endl; 
    std::cout << "dimBlock: " << dimBlock.x << " " << dimBlock.y << " " << dimBlock.z << std::endl; 
    

    getGrayscaleImageKernel<<<dimGrid, dimBlock>>>(image_d, grayscaleImage_d, imageWidth, imageHeight);

    cudaMemcpy(grayscaleImage_h, grayscaleImage_d, giSize, cudaMemcpyDeviceToHost);

    cudaFree(image_d);
    cudaFree(grayscaleImage_h);
}

void saveGrayscaleImage(int* grayscaleImage, std::string fileName){
    std::ofstream grayImageFile;
    grayImageFile.open ( fileName + ".pgm");

    grayImageFile << "P2\n" << imageWidth << " " << imageHeight << "\n" << 255 << "\n";

    for (size_t j = 0; j < imageHeight; j++)
    {
        for (size_t i = 0; i < imageWidth; i++)
        {
            grayImageFile << grayscaleImage[j* imageWidth + i] << " ";
        }
        grayImageFile << "\n";
    }
    
    grayImageFile.close();
}

int main() {
    std::string fileName = "sample.ppm"; 
    int* image_h = getPPMImageFromFile(fileName);
    int* grayScaleImage_h = (int*) malloc( imageHeight * imageWidth * sizeof(int));
    
    getGrayscaleImage(image_h, grayScaleImage_h);
    saveGrayscaleImage(grayScaleImage_h, fileName);
    

    free(image_h);
    free(grayScaleImage_h);
    return 0;
}