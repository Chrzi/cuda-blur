#include <iostream>
#include <opencv2/opencv.hpp>
#include "cmath"
#include "blur.cuh"

using namespace std;
using namespace cv;

#define errCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"[ERROR]: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


// This function takes a linearized matrix in the form of a vector and
// calculates elements according to the 2D Gaussian distribution
void generateGaussian(float* kernel, int dim) {
    int radius = dim / 2;
    float stdev = 2.0;
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


int main(int argc, char** argv) {
    string imageLocation = "image.jpg";

    if (argc == 2) {
        imageLocation = imageLocation.assign(argv[1]);
    } else if (argc > 2) {
        printf("Usage: gaussBlur [image]\n");
        return EXIT_FAILURE;
    }

    Mat image = cv::imread(imageLocation, IMREAD_COLOR);
    if (image.empty()) {
        printf("Could not read image!");
        return EXIT_FAILURE;
    }
    // convert image into 8 bit and split channels bgr
    Mat image8 = Mat();
    image.convertTo(image8, CV_8U);
    vector<Mat> input(3);
    split(image8, input);

    auto size = sizeof(unsigned char) * input[0].total();
    int height = input[0].rows;
    int width = input[0].cols;
    unsigned char* input_copy[3];

    for (int i = 0; i < 3; ++i) {
        errCheck(cudaMallocHost(&(input_copy[i]), size));
        memcpy(input_copy[i], input[i].data, size);
    }

    unsigned char* output[3];
    for (int i = 0; i < 3; ++i) {
        errCheck(cudaMallocHost(&(output[i]), size));
    }


    // generate and copy kernel
    int kernelDim = 7;
    float kernel[kernelDim * kernelDim];
    generateGaussian(kernel, kernelDim);
    float* kernelGPU;
    errCheck(cudaMalloc(&kernelGPU, sizeof(float) * kernelDim * kernelDim))
    errCheck(cudaMemcpy(kernelGPU, kernel, sizeof(float) * kernelDim * kernelDim , cudaMemcpyHostToDevice));


    //init 3 streams
    cudaStream_t stream_B, stream_G, stream_R;
    cudaStreamCreate(&stream_B);
    cudaStreamCreate(&stream_G);
    cudaStreamCreate(&stream_R);
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
            width / threadsPerBlock.x,
            height / threadsPerBlock.y
            );

    // allocate memory
    unsigned char* input_b;
    unsigned char* input_g;
    unsigned char* input_r;
    errCheck(cudaMallocAsync(&input_b, size, stream_B));
    errCheck(cudaMallocAsync(&input_g, size, stream_G));
    errCheck(cudaMallocAsync(&input_r, size, stream_R));
    errCheck(cudaMemcpyAsync(input_b, input_copy[0], size, cudaMemcpyHostToDevice, stream_B));
    errCheck(cudaMemcpyAsync(input_g, input_copy[1], size, cudaMemcpyHostToDevice, stream_G));
    errCheck(cudaMemcpyAsync(input_r, input_copy[2], size, cudaMemcpyHostToDevice, stream_R));

    unsigned char* output_b;
    unsigned char* output_g;
    unsigned char* output_r;
    errCheck(cudaMallocAsync(&output_b, size, stream_B));
    errCheck(cudaMallocAsync(&output_g, size, stream_G));
    errCheck(cudaMallocAsync(&output_r, size, stream_R));

    unsigned char* output_host_b;
    unsigned char* output_host_g;
    unsigned char* output_host_r;
    errCheck(cudaMallocHost(&output_host_b, size));
    errCheck(cudaMallocHost(&output_host_g, size));
    errCheck(cudaMallocHost(&output_host_r, size));

    Gaussian<<<numBlocks, threadsPerBlock, 0, stream_B>>>(input_b, output_b, width, height, kernelGPU, kernelDim);
    Gaussian<<<numBlocks, threadsPerBlock, 0, stream_G>>>(input_g, output_g, width, height, kernelGPU, kernelDim);
    Gaussian<<<numBlocks, threadsPerBlock, 0, stream_R>>>(input_r, output_r, width, height, kernelGPU, kernelDim);
    errCheck(cudaDeviceSynchronize());

    //copy back to Host memory
    cudaMemcpy(output_host_b, output_b, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(output_host_g, output_g, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(output_host_r, output_r, size, cudaMemcpyDeviceToHost);
    errCheck(cudaDeviceSynchronize());

    vector<Mat> output_v(3);
    output_v[0] = Mat(height, width, CV_8UC1, output_host_b);
    output_v[1] = Mat(height, width, CV_8UC1, output_host_g);
    output_v[2] = Mat(height, width, CV_8UC1, output_host_r);

    Mat outputImg = Mat();
    merge(output_v, outputImg);

//    cv::namedWindow("Image", WINDOW_GUI_NORMAL);
//    cv::imshow("Image", image8);
//    namedWindow("Output", WINDOW_GUI_NORMAL);
//    imshow("Output", outputImg);
//    cv::namedWindow("Image Blue", WINDOW_GUI_NORMAL);
//    cv::imshow("Image Blue", input[0]);
//    cv::namedWindow("Image Green", WINDOW_GUI_NORMAL);
//    cv::imshow("Image Green", input[1]);
//    cv::namedWindow("Image Red", WINDOW_GUI_NORMAL);
//    cv::imshow("Image Red", input[2]);

//    waitKey(0);

    imwrite("output.jpg", outputImg);
    imwrite("output_blue.jpg", output_v[0]);
    imwrite("output_green.jpg", output_v[1]);
    imwrite("output_red.jpg", output_v[2]);

    return EXIT_SUCCESS;
}

