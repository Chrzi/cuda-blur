//
// Created by christian on 6/15/22.
//

#ifndef GAUSSBLUR_BLUR_CUH
#define GAUSSBLUR_BLUR_CUH

__global__ void Gaussian(const unsigned char *input, unsigned char *output, const unsigned int width, const unsigned int height, const float *kernel, const unsigned int kernelWidth);


#endif //GAUSSBLUR_BLUR_CUH
