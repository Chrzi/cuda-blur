cmake_minimum_required(VERSION 3.18)
project(gaussBlur CUDA C CXX)

set(CMAKE_CUDA_STANDARD 14)
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(gaussBlur main.cu blur.cu blur.cuh)


set_target_properties(gaussBlur PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)
target_link_libraries( gaussBlur ${OpenCV_LIBS} )
