#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#endif
#include <algorithm>

__global__ void dialsDijkstra3D_CUDA_Kernel(
    int* dist,
    int* bucket_flags,
    int* next_bucket_flags,
    int* obstacles_sorted,
    int num_obstacles,
    int N, int M, int K,
    int current_distance,
    bool* updated) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_voxels = N * M * K;
    
    if (idx >= total_voxels) return;
    
    if (dist[idx] != current_distance || bucket_flags[idx] == 0) return;
    
    int x = idx / (M * K);
    int y = (idx % (M * K)) / K;
    int z = idx % K;
    
    const int dx[] = {-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    const int dy[] = {-1,-1,-1, 0, 0, 0, 1, 1, 1,-1,-1,-1, 0, 0, 1, 1, 1,-1,-1,-1, 0, 0, 0, 1, 1, 1};
    const int dz[] = {-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1};
    const int weights[] = {17,14,17,14,10,14,17,14,17,14,10,14,10,10,14,10,14,17,14,17,14,10,14,17,14,17};
    
    for (int n = 0; n < 26; n++) {
        int nx = x + dx[n];
        int ny = y + dy[n];
        int nz = z + dz[n];
        
        if (nx < 0 || nx >= N || ny < 0 || ny >= M || nz < 0 || nz >= K) continue;
        
        int neighbor_idx = nx * M * K + ny * K + nz;
        
        bool is_obstacle = false;
        int left = 0, right = num_obstacles - 1;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (obstacles_sorted[mid] == neighbor_idx) {
                is_obstacle = true;
                break;
            } else if (obstacles_sorted[mid] < neighbor_idx) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        if (is_obstacle) continue;
        
        int new_dist = current_distance + weights[n];
        
        int old_dist = atomicMin(&dist[neighbor_idx], new_dist);
        
        if (new_dist < old_dist) {
            next_bucket_flags[neighbor_idx] = 1;
            *updated = true;
        }
    }
}

std::vector<int> dialsDijkstra3D_CUDA(const std::vector<int>& sources, 
                                     const std::vector<int>& obstacle_indices,
                                     int N, int M, int K) {
    const int INF = std::numeric_limits<int>::max();
    int total_voxels = N * M * K;
    
    std::vector<int> obstacles_sorted = obstacle_indices;
    std::sort(obstacles_sorted.begin(), obstacles_sorted.end());
    
    thrust::device_vector<int> d_dist(total_voxels, INF);
    thrust::device_vector<int> d_bucket_flags(total_voxels, 0);
    thrust::device_vector<int> d_next_bucket_flags(total_voxels, 0);
    thrust::device_vector<int> d_obstacles(obstacles_sorted);
    thrust::device_vector<bool> d_updated(1, false);
    
    for (int src : sources) {
        d_dist[src] = 0;
        d_bucket_flags[src] = 1;
    }
    
    int block_size = 256;
    int grid_size = (total_voxels + block_size - 1) / block_size;
    
    for (int current_distance = 0; current_distance < 17000; current_distance++) {
        thrust::fill(d_updated.begin(), d_updated.end(), false);
        
        dialsDijkstra3D_CUDA_Kernel<<<grid_size, block_size>>>(
            thrust::raw_pointer_cast(d_dist.data()),
            thrust::raw_pointer_cast(d_bucket_flags.data()),
            thrust::raw_pointer_cast(d_next_bucket_flags.data()),
            thrust::raw_pointer_cast(d_obstacles.data()),
            obstacles_sorted.size(),
            N, M, K,
            current_distance,
            thrust::raw_pointer_cast(d_updated.data())
        );
        
        cudaDeviceSynchronize();
        
        bool updated;
        thrust::copy(d_updated.begin(), d_updated.end(), &updated);
        if (!updated) break;
        
        thrust::fill(d_bucket_flags.begin(), d_bucket_flags.end(), 0);
        thrust::copy(d_next_bucket_flags.begin(), d_next_bucket_flags.end(), d_bucket_flags.begin());
        thrust::fill(d_next_bucket_flags.begin(), d_next_bucket_flags.end(), 0);
    }
    
    std::vector<int> result(total_voxels);
    thrust::copy(d_dist.begin(), d_dist.end(), result.begin());
    
    return result;
}