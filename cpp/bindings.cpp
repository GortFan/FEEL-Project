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

int index2D(int x, int y, int z, int M, int K);
std::vector<std::vector<std::pair<int, int>>> makeAdjMatrix2D(int N, int M, int K, const std::vector<int>& obstacle_indices);
std::vector<int> dialsDijkstra2D(const std::vector<std::vector<std::pair<int, int>>>& adj, const std::vector<int>& sources, int N, int M, int K);
// bool isObstacle2D(int idx, const std::unordered_set<int>& obstacles);

int index3D(int x, int y, int z, int M, int K);
std::vector<std::vector<std::pair<int, int>>> makeAdjMatrix3D(int N, int M, int K, const std::vector<int>& obstacle_indices);
std::vector<int> dialsDijkstra3D(const std::vector<std::vector<std::pair<int, int>>>& adj, const std::vector<int>& sources, int N, int M, int K);
// bool isObstacle3D(int idx, const std::unordered_set<int>& obstacles)

std::vector<int> dialsDijkstra3D_Implicit(const std::vector<int>& sources, const std::vector<int>& obstacle_indices, int N, int M, int K);

PYBIND11_MODULE(mybindings, m) {
    m.def("add", &add, "A function that adds two numbers");
    m.def("subtract", &subtract, "A function that subtracts two numbers");

    m.def("index", &index2D, "A function that converts matrix coordinates to flattened array index values");
    m.def("makeAdjMatrix", &makeAdjMatrix2D, "Creates an adjacency matrix for a given 2D matrix dimension");
    m.def("dialsDijkstra", &dialsDijkstra2D, "Performs dials algorithm on the 2D adjacency matrix, given a set of source coordinates");
    // m.def("isObstacle2D", &isObstacle2D, "Checks if a flattened array index corresponding to matrix cell is 'impermeable' by comparing it against a set of obstacle indices.");
    
    m.def("index3D", &index3D, "A function that converts 3D matrix coordinates to flattened array index values");
    m.def("makeAdjMatrix3D", &makeAdjMatrix3D, "Creates an adjacency matrix for a given 3D matrix dimension");
    m.def("dialsDijkstra3D", &dialsDijkstra3D, "Performs dials algorithm on the 3D adjacency matrix, given a set of source coordinates");
    // m.def("isObstacle3D", &isObstacle3D, "Checks if a flattened array index corresponding to matrix cell is 'impermeable' by comparing it against a set of obstacle indices.");

    m.def("dialsDijkstra3D_Implicit", &dialsDijkstra3D_Implicit, "Performs dials algorithm on an implicitly defined matrix assuming Moore Neighborhood, requires matrix dimensions, a set of source indices, and obstacle indices");
}
