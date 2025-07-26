cmake_minimum_required(VERSION 3.21)
project(mybindings)

set(CMAKE_CXX_STANDARD 17)

# Point CMake to your pybind11 install
set(pybind11_DIR "D:/pybind11-install/share/cmake/pybind11")

find_package(pybind11 REQUIRED)

file(GLOB SOURCES "cpp/*.cpp")

pybind11_add_module(mybindings ${SOURCES})
