#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <vector>
#include <utility>

// Declare function implemented in another file
int add(int i, int j);
int subtract(int i, int j);
int index(int x, int y, int z, int M, int K);
std::vector<std::vector<std::pair<int, int>>> makeAdjMatrix(int N, int M, int K, const std::vector<int>& obstacle_indices);
std::vector<int> dialsDijkstra(const std::vector<std::vector<std::pair<int, int>>>& adj, const std::vector<int>& sources, int N, int M, int K);

PYBIND11_MODULE(mybindings, m) {
    m.def("add", &add, "A function that adds two numbers");
    m.def("subtract", &subtract, "A function that subtracts two numbers");
    m.def("index", &index, "A function that converts matrix coordinates to flattened array index values");
    m.def("makeAdjMatrix", &makeAdjMatrix, "Creates an adjacency matrix for a given matrix dimension");
    m.def("dialsDijkstra", &dialsDijkstra, "Performs dials algorithm on the adjacency matrix, given a set of source coordinates");
}
