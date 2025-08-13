cmake_minimum_required(VERSION 3.21)
project(FEELcpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Point CMake to your pybind11 install
set(pybind11_DIR "D:/pybind11-install/share/cmake/pybind11")

find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Include BOTH .cpp and .cu files
file(GLOB SOURCES "cpp/*.cpp" "cpp/*.cu")

pybind11_add_module(FEELcpp ${SOURCES})

# Set CUDA properties
set_property(TARGET FEELcpp PROPERTY CUDA_STANDARD 17)
set_property(TARGET FEELcpp PROPERTY CUDA_ARCHITECTURES native)

# Use static linking to avoid DLL dependencies
target_link_libraries(FEELcpp PRIVATE CUDA::cudart_static)

target_compile_definitions(FEELcpp PRIVATE VERSION_INFO=${whar})