#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <vector>
#include <utility>


int index2D(int x, int y, int z, int M, int K);
std::vector<std::vector<std::pair<int, int>>> makeAdjMatrix2D(int N, int M, int K, const std::vector<int>& obstacle_indices);
std::vector<int> dialsDijkstra2D(const std::vector<std::vector<std::pair<int, int>>>& adj, const std::vector<int>& sources, int N, int M, int K);

int index3D(int x, int y, int z, int M, int K);
std::vector<std::vector<std::pair<int, int>>> makeAdjMatrix3D(int N, int M, int K, const std::vector<int>& obstacle_indices);
std::vector<int> dialsDijkstra3D(const std::vector<std::vector<std::pair<int, int>>>& adj, const std::vector<int>& sources, int N, int M, int K);

std::vector<int> dialsDijkstra3D_Implicit(const std::vector<int>& sources, const std::vector<int>& obstacle_indices, int N, int M, int K);

std::vector<int> dialsDijkstra3D_CUDA(const std::vector<int>& sources, const std::vector<int>& obstacle_indices, int N, int M, int K);

PYBIND11_MODULE(FEELcpp, m) {
    m.def("index", &index2D, "A function that converts matrix coordinates to flattened array index values");
    m.def("makeAdjMatrix", &makeAdjMatrix2D, "Creates an adjacency matrix for a given 2D matrix dimension");
    m.def("dialsDijkstra", &dialsDijkstra2D, "Performs dials algorithm on the 2D adjacency matrix, given a set of source coordinates");
    
    m.def("index3D", &index3D, "A function that converts 3D matrix coordinates to flattened array index values");
    m.def("makeAdjMatrix3D", &makeAdjMatrix3D, "Creates an adjacency matrix for a given 3D matrix dimension");
    m.def("dialsDijkstra3D", &dialsDijkstra3D, "Performs dials algorithm on the 3D adjacency matrix, given a set of source coordinates");

    m.def("dialsDijkstra3D_Implicit", &dialsDijkstra3D_Implicit, "Performs dials algorithm on an implicitly defined matrix assuming Moore Neighborhood, requires matrix dimensions, a set of source indices, and obstacle indices");
    m.def("dialsDijkstra3D_CUDA", &dialsDijkstra3D_CUDA, "CUDA Version of dialsDijkstra3D_Implicit - Performs dials algorithm on an implicitly defined matrix assuming Moore Neighborhood, requires matrix dimensions, a set of source indices, and obstacle indices");
}
