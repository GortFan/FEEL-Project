#include <iostream>
#include <vector>
#include <deque>
#include <random>
#include <limits>
#include <chrono>
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#include <omp.h>
#include <pybind11/pybind11.h>

using namespace std;

inline bool is_obstacle3D(int idx, const std::unordered_set<int>& obstacles) {
    return obstacles.find(idx) != obstacles.end();
}

std::vector<std::vector<std::pair<int, int>>> makeAdjMatrix3D(
    int N, int M, int K,
    const std::vector<int>& obstacle_indices
) {
    std::unordered_set<int> obstacles(obstacle_indices.begin(), obstacle_indices.end());
    auto index = [M, K](int x, int y, int z) {
        return x * M * K + y * K + z;
    };
    int S = N * M * K;
    std::vector<std::vector<std::pair<int, int>>> adj(S);

    for (int x = 0; x < N; ++x) {
        for (int y = 0; y < M; ++y) {
            for (int z = 0; z < K; ++z) {
                int u = index(x, y, z);
                if (obstacles.count(u)) continue; // skip obstacle nodes

                // Moore neighborhood: all 26 neighbors (excluding self)
                for (int dx = -1; dx <= 1; ++dx) {
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dz = -1; dz <= 1; ++dz) {
                            if (dx == 0 && dy == 0 && dz == 0) continue; // skip self
                            int nx = x + dx, ny = y + dy, nz = z + dz;
                            if (nx < 0 || nx >= N || ny < 0 || ny >= M || nz < 0 || nz >= K)
                                continue;
                            int v = index(nx, ny, nz);
                            if (obstacles.count(v)) continue; // skip obstacles
                            int weight = static_cast<int>(10 * std::sqrt(dx*dx + dy*dy + dz*dz));
                            adj[u].emplace_back(v, weight);
                        }
                    }
                }
            }
        }
    }
    return adj;
}

// vector<int> generateSources(int count, int size) {
//     vector<int> sources;
//     random_device rd;
//     mt19937 gen(rd());
//     uniform_int_distribution<> dis(0, size - 1);
//     for (int i = 0; i < count; ++i)
//         sources.push_back(dis(gen));
//     return sources;
// }

int index3D(int x, int y, int z, int M, int K) {
    return x * M * K + y * K + z;
}

vector<int> dialsDijkstra3D(const vector<vector<pair<int, int>>>& adj, const vector<int>& sources, int N, int M, int K) {
    const int INF = numeric_limits<int>::max();
    vector<int> dist(N * M * K, INF);
    const int MAX_EDGE_WEIGHT = 17;
    int maxDist = MAX_EDGE_WEIGHT * 1000;

    vector<deque<int>> buckets(maxDist + 1);
    int currentBucket = 0;

    for (int src : sources) {
        dist[src] = 0;
        buckets[0].push_back(src);
    }

    while (currentBucket <= maxDist) {
        while (currentBucket <= maxDist && buckets[currentBucket].empty())
            ++currentBucket;
        if (currentBucket > maxDist) break;

        int u = buckets[currentBucket].front();
        buckets[currentBucket].pop_front();

        if (dist[u] < currentBucket) continue;

        for (auto [v, w] : adj[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                buckets[dist[v]].push_back(v);
            }
        }
    }

    return dist;
}