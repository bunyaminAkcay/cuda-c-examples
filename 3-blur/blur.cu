#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <fstream>

int blockSizeW = 16;
int blurSize = 5;
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
void blurKernel(int* image_d, int* blurImage_d, int imageW, int imageH, int blurSize){
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < imageW && row < imageH)
    {
        int pixelValueR = 0, pixelValueG = 0, pixelValueB = 0;
        int pixelCount = 0;

        for (int blurOffsetRow = -blurSize; blurOffsetRow < blurSize; blurOffsetRow++)
        {
            for (int blurOffsetCol = -blurSize; blurOffsetCol < blurSize; blurOffsetCol++)
            {
                int currPixelCol = col + blurOffsetCol;
                int currPixelRow = row + blurOffsetRow;

                if (currPixelCol >=0 && currPixelCol < imageW && currPixelRow >=0 && currPixelRow < imageH)
                {
                    pixelValueR += image_d[(currPixelRow * imageW + currPixelCol) * 3    ];
                    pixelValueG += image_d[(currPixelRow * imageW + currPixelCol) * 3 + 1];
                    pixelValueB += image_d[(currPixelRow * imageW + currPixelCol) * 3 + 2];
                    pixelCount++;
                }
            }
        }
        blurImage_d[(row * imageW + col) * 3    ] = pixelValueR / pixelCount;
        blurImage_d[(row * imageW + col) * 3 + 1] = pixelValueG / pixelCount;
        blurImage_d[(row * imageW + col) * 3 + 2] = pixelValueB / pixelCount;
    }
}

void blur(int* image_h, int* blurImage_h){
    int *image_d, *blurImage_d;

    int size = imageHeight * imageWidth * 3 * sizeof(int);

    cudaMalloc((void **) &image_d, size);
    cudaMalloc((void **) &blurImage_d, size);

    cudaMemcpy(image_d, image_h, size, cudaMemcpyHostToDevice);

    dim3 dimBlock( blockSizeW, blockSizeW, 1);
    dim3 dimGrid( (imageWidth + blockSizeW -1 )/blockSizeW, (imageHeight + blockSizeW -1 )/blockSizeW, 1 );

    blurKernel<<<dimGrid, dimBlock>>>(image_d, blurImage_d, imageWidth, imageHeight, blurSize);

    cudaMemcpy(blurImage_h, blurImage_d, size, cudaMemcpyDeviceToHost);

    cudaFree(image_d);
    cudaFree(blurImage_d);
}

void savePPM3Image(int* ppm3Image, std::string fileName){
    std::ofstream ppm3ImageFile;
    ppm3ImageFile.open ( fileName + ".ppm");

    ppm3ImageFile << "P3\n" << imageWidth << " " << imageHeight << "\n" << 255 << "\n";

    for (size_t j = 0; j < imageHeight; j++)
    {
        for (size_t i = 0; i < imageWidth; i++)
        {
            ppm3ImageFile << ppm3Image[3 * (j* imageWidth + i)] << " " << ppm3Image[3 * (j* imageWidth + i) + 1] << " " << ppm3Image[3 * (j* imageWidth + i) + 2] << " ";
        }
        ppm3ImageFile << "\n";
    }
    
    ppm3ImageFile.close();
}

int main() {
    std::string fileName = "sample.ppm"; 
    int* image_h = getPPMImageFromFile(fileName);
    int* blurImage_h = (int*) malloc( imageHeight * imageWidth * 3 * sizeof(int));
    
    blur(image_h, blurImage_h);
    savePPM3Image(blurImage_h, fileName);

    free(image_h);
    free(blurImage_h);
    return 0;
}