//
// Created by christian on 6/15/22.
//

#include "blur.cuh"

__global__ void Gaussian(const unsigned char *input, unsigned char *output, const unsigned int width, const unsigned int height, const float *kernel, const unsigned int kernelWidth) {

    unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (row < height && col < width) {
        int radius = kernelWidth / 2;
        float sum = 0.0;
        for (int i = -radius; i <= radius; i++) {
            for (int j = -radius; j <= radius; j++) {

                unsigned int y = max(0, min(height - 1, row + i));
                unsigned int x = max(0, min(width - 1, col + j));

                float weight = kernel[(j + radius) + (i + radius) * kernelWidth];
                sum += weight * input[x + y * width];
            }
        }

        output[col + row * width] = static_cast<unsigned char>(sum);
    }
}

