# Gaussian Blur

## How to build and execute

0. Cuda, nvcc and a recent C compiler are already installed
1. Install OpenCV `apt-get update && apt-get install libopencv-dev`
2. Install cmake `apt-get install cmake`
3. Create a build directory inside the project direcotry `mkdir build && cd build`
4. Create build plan with CMake `cmake ../`
5. Build the project using the build plan `cmake --build .`
6. Execute the program `./gaussBlur [Image-Path]` it defaults to using the file `image.jpg` if no other path is given.

## Things to tweak

- **Kernel Size** (Size of the Matrix) in main.cu `#define KERNEL_SIZE 5` (must be an odd number!)
- **Sigma** in main.cu `#define SIGMA 2.0` (the bigger the sigma the higher the weight of surrounding pixels)
- **GUI** in main.cu `#define GUI`, en-/disable GUI by (un-)commenting the line!