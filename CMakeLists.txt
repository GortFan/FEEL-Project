cmake_minimum_required(VERSION 3.21)
project(FEELcpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(pybind11_DIR "D:/pybind11-install/share/cmake/pybind11")

find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)

file(GLOB SOURCES "cpp/*.cpp" "cpp/*.cu")

pybind11_add_module(FEELcpp ${SOURCES})

set_property(TARGET FEELcpp PROPERTY CUDA_STANDARD 17)
set_property(TARGET FEELcpp PROPERTY CUDA_ARCHITECTURES native)

target_link_libraries(FEELcpp PRIVATE CUDA::cudart_static)

target_compile_definitions(FEELcpp PRIVATE VERSION_INFO=${whar})