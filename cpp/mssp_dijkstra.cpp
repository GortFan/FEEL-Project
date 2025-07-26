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

using namespace std;

const int N = 100, M = 100, K = 100;
const int S = N * M * K;
const int INF = numeric_limits<int>::max();
const int MAX_EDGE_WEIGHT = 17;
const int SCALE = 10; // weight multiplier

size_t getCurrentRSS() {
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return static_cast<size_t>(info.WorkingSetSize);
}

inline int index(int x, int y, int z) {
    return x * M * K + y * K + z;
}

vector<int> generateSources(int count, int size) {
    vector<int> sources;
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, size - 1);
    for (int i = 0; i < count; ++i)
        sources.push_back(dis(gen));
    return sources;
}

vector<int> dialsDijkstra(const vector<vector<pair<int, int>>>& adj, const vector<int>& sources) {
    vector<int> dist(S, INF);
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

int main() {
    size_t mem_before = getCurrentRSS();
    auto start_adj = chrono::high_resolution_clock::now();

    vector<vector<pair<int, int>>> adj(S);
    vector<tuple<int, int, int>> deltas = {
        {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1},
        {1, 1, 0}, {1, -1, 0}, {-1, 1, 0}, {-1, -1, 0},
        {1, 0, 1}, {1, 0, -1}, {-1, 0, 1}, {-1, 0, -1},
        {0, 1, 1}, {0, 1, -1}, {0, -1, 1}, {0, -1, -1},
        {1, 1, 1}, {-1, 1, 1}, {1, -1, 1}, {-1, -1, 1},
        {1, 1, -1}, {-1, 1, -1}, {1, -1, -1}, {-1, -1, -1}
    };
    vector<int> weights = {
        10, 10, 10, 10, 10, 10,
        14, 14, 14, 14,
        14, 14, 14, 14,
        14, 14, 14, 14,
        17, 17, 17, 17, 17, 17, 17, 17
    };

    #pragma omp parallel for collapse(3)
    for (int x = 0; x < N; ++x) {
        for (int y = 0; y < M; ++y) {
            for (int z = 0; z < K; ++z) {
                int u = index(x, y, z);
                vector<pair<int, int>> local_edges;
                for (size_t i = 0; i < deltas.size(); ++i) {
                    auto [dx, dy, dz] = deltas[i];
                    int xi = x + dx;
                    int yi = y + dy;
                    int zi = z + dz;
                    if (xi >= 0 && xi < N && yi >= 0 && yi < M && zi >= 0 && zi < K) {
                        int v = index(xi, yi, zi);
                        local_edges.emplace_back(v, weights[i]);
                    }
                }
                #pragma omp critical
                adj[u] = move(local_edges);
            }
        }
    }

    auto end_adj = chrono::high_resolution_clock::now();
    chrono::duration<double> adj_time = end_adj - start_adj;
    cout << "Adjacency build time: " << adj_time.count() << " seconds\n";

    auto start_dijkstra = chrono::high_resolution_clock::now();
    vector<int> sources = generateSources(100, S);
    vector<int> dist = dialsDijkstra(adj, sources);
    auto end_dijkstra = chrono::high_resolution_clock::now();

    for (int i = 0; i < 10; ++i) {
        cout << "Distance to node " << i << ": " << dist[i] / (double)SCALE << endl;
    }

    chrono::duration<double> dijkstra_time = end_dijkstra - start_dijkstra;
    size_t mem_after = getCurrentRSS();

    cout << "Dijkstra time:  " << dijkstra_time.count() << " seconds\n";
    cout << "Memory before: " << mem_before / (1024.0 * 1024.0) << " MB\n";
    cout << "Memory after:  " << mem_after  / (1024.0 * 1024.0) << " MB\n";
    cout << "Memory delta:  " << (mem_after - mem_before) / (1024.0 * 1024.0) << " MB\n";

    return 0;
}
