cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
cmake_policy(SET CMP0054 NEW)
project(HSegBenchmark)

add_definitions(-D__MSC_VER__)

set(Torch_DIR "D:/Library/Cpp/CV/libtorch/share/cmake/Torch")
set(TorchVision_DIR "D:/Library/Cpp/CV/torchvision/share/cmake/TorchVision")
set(OpenCV_DIR "D:/Library/Cpp/CV/opencv/build")
find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(D:/Library/Cpp/CV/libtorch/include)
include_directories(D:/Library/Cpp/CV/libtorch/include/torch/csrc/api/include)
include_directories(D:/Library/Cpp/CV/torchvision/include)
include_directories(D:/Library/Cpp/CV/opencv/build/include)
include_directories(./include)

set(WITH_GTK false)
set(OPENCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)

add_executable(HSegBenchmark main.cpp processing.cpp processing.h include/_torch/torch.h)
target_link_libraries(HSegBenchmark "${TORCH_LIBRARIES}")
target_link_libraries(HSegBenchmark "${TORCHVISION_LIBRARIES}")
target_link_libraries(HSegBenchmark "${OPENCV_LIBS}")

set_property(TARGET HSegBenchmark PROPERTY CXX_STANDARD 17)

