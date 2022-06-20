#include <iostream>
#include <opencv2/opencv.hpp>
#include "cmath"
#include "blur.cuh"

using namespace std;
using namespace cv;

#define KERNEL_SIZE 5
#define SIGMA 2.0
#define GUI

#define errCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "[ERROR]: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


// This function takes a linearized matrix in the form of a vector and
// calculates elements according to the 2D Gaussian distribution
void generateGaussian(float *kernel, int dim) {
    int radius = dim / 2;
    float stdev = SIGMA;
    float constant = 1.0 / (2.0 * M_PI * pow(stdev, 2));
    float sum = 0.0;

    for (int i = -radius; i < radius + 1; ++i) {
        for (int j = -radius; j < radius + 1; ++j) {
            kernel[(i + radius) * dim + (j + radius)] =
                    constant * (1 / exp(i * i + j * j) / (2 * pow(stdev, 2)));
            sum += kernel[(i + radius) * dim + (j + radius)];
        }
    }

    //normalize
    for (int i = -radius; i < radius + 1; ++i) {
        for (int j = -radius; j < radius + 1; ++j) {
            kernel[(i + radius) * dim + (j + radius)] /= sum;
        }
    }
}

void displayImage(string name, Mat image) {
    namedWindow(name, WINDOW_GUI_EXPANDED);
    imshow(name, image);
}

int main(int argc, char **argv) {
    string imageLocation = "image.jpg";

    if (argc == 2) {
        imageLocation = imageLocation.assign(argv[1]);
    } else if (argc > 2) {
        printf("Usage: gaussBlur [image]\n");
        return EXIT_FAILURE;
    }

    Mat image = imread(imageLocation, IMREAD_COLOR);
    if (image.empty()) {
        printf("Could not read image!");
        return EXIT_FAILURE;
    }
    // convert image into 8 bit and split channels (BGR)
    Mat image8 = Mat();
    image.convertTo(image8, CV_8U);
    vector<Mat> inputSplit(3);
    split(image8, inputSplit);

    auto size = sizeof(unsigned char) * inputSplit[0].total();
    int height = inputSplit[0].rows;
    int width = inputSplit[0].cols;

    //allocate pinned host memory for input and output
    unsigned char *inputHost[3];
    unsigned char *outputHost[3];
    for (int i = 0; i < 3; ++i) {
        errCheck(cudaMallocHost(&(inputHost[i]), size));
        memcpy(inputHost[i], inputSplit[i].data, size);
        errCheck(cudaMallocHost(&(outputHost[i]), size));
    }

    // generate and copy kernel
    float kernel[KERNEL_SIZE * KERNEL_SIZE];
    generateGaussian(kernel, KERNEL_SIZE);
    float *kernelGPU;
    errCheck(cudaMalloc(&kernelGPU, sizeof(float) * KERNEL_SIZE * KERNEL_SIZE))
    errCheck(cudaMemcpy(kernelGPU, kernel, sizeof(float) * KERNEL_SIZE * KERNEL_SIZE, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
            width / threadsPerBlock.x,
            height / threadsPerBlock.y
    );

    //init 3 streams
    cudaStream_t streams[3];
    unsigned char *inputChannels[3];
    unsigned char *outputChannels[3];
    for (int i = 0; i < 3; ++i) {
        //create streams
        cudaStreamCreate(&streams[i]);

        //alloc inputSplit and output memory on device
        errCheck(cudaMallocAsync(&inputChannels[i], size, streams[i]));
        errCheck(cudaMallocAsync(&outputChannels[i], size, streams[i]));

        //copy inputSplit onto device
        errCheck(cudaMemcpyAsync(
                inputChannels[i], inputHost[i], size, cudaMemcpyHostToDevice, streams[i]
        ));

        //call kernel
        Gaussian<<<numBlocks, threadsPerBlock, 0, streams[0]>>>(
                inputChannels[i], outputChannels[i], width, height, kernelGPU, KERNEL_SIZE);
    }
    errCheck(cudaDeviceSynchronize());

    for (int i = 0; i < 3; ++i) {
        //copy back to Host memory
        cudaMemcpy(outputHost[i], outputChannels[i], size, cudaMemcpyDeviceToHost);
    }
    errCheck(cudaDeviceSynchronize());

    for (int i = 0; i < 3; ++i) {
        cudaFreeHost(inputHost[i]);
        cudaFree(inputChannels[i]);
        cudaFree(outputChannels[i]);
    }

    vector<Mat> outputSplit(3);
    for (int i = 0; i < 3; ++i) {
        outputSplit[i] = Mat(height, width, CV_8UC1, outputHost[i]);
    }
    Mat outputImg = Mat();
    merge(outputSplit, outputImg);

#ifdef GUI
    displayImage("Original", image8);
    displayImage("Output", outputImg);
    displayImage("Original Blue", inputSplit[0]);
    displayImage("Original Green", inputSplit[1]);
    displayImage("Original Red", inputSplit[2]);
    waitKey(0);
#endif

    imwrite("output.jpg", outputImg);
    imwrite("output_blue.jpg", outputSplit[0]);
    imwrite("output_green.jpg", outputSplit[1]);
    imwrite("output_red.jpg", outputSplit[2]);

    return EXIT_SUCCESS;
}

