//
// Created by christian on 6/15/22.
//

#include "blur.cuh"

__global__ void Gaussian(const unsigned char *input, unsigned char *output, const unsigned int width, const unsigned int height, const float *kernel, const unsigned int kernelWidth) {

    unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (row < height && col < width) {
        const int half = kernelWidth / 2;
        float sum = 0.0;
        for (int i = -half; i <= half; i++) {
            for (int j = -half; j <= half; j++) {

                unsigned int y = max(0, min(height - 1, row + i));
                unsigned int x = max(0, min(width - 1, col + j));

                float weight = kernel[(j + half) + (i + half) * kernelWidth];
                sum += weight * input[x + y * width];
            }
        }

        output[col + row * width] = static_cast<unsigned char>(sum);
    }
}

