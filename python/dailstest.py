import os
import sys
import numpy as np

project_root = os.path.dirname(os.path.dirname(__file__))

sys.path.append(os.path.join(project_root, 'build', 'Release'))

import mybindings as bindings

m = np.ones((4,3,3))
m[1,1,1] = 0
N, M, K = m.shape
max_int = 2147483647 # from std::numeric_limits<int>::max(), for obstacle nodes

def get_obstacles_indices(m):
    """Returns a list of obstacles (0=obstacle, 1=traversible) in 1D array indexing style"""
    obstacle_positions_3D = np.where(m == 0)
    print(obstacle_positions_3D)
    _, M, K = m.shape
    obstacle_positions_1D = (obstacle_positions_3D[0] * M * K +
                            obstacle_positions_3D[1] * K +
                            obstacle_positions_3D[2])
    print(obstacle_positions_1D.tolist())
    return obstacle_positions_1D.tolist()

def generate_sources(m):
    """Returns a list of source nodes in 1D array indexing style"""
    N, M, K = m.shape
    
    last_layer = m[N-1, :, :]
    source_positions_2d = np.where(last_layer == 1)
    
    y_coords = source_positions_2d[0]
    z_coords = source_positions_2d[1]
    x_coords = np.full_like(y_coords, N-1)
    
    sources = (x_coords * M * K + y_coords * K + z_coords).tolist()
    return sources 

sources = generate_sources(m)

dt = bindings.dialsDijkstra3D_Implicit(sources, get_obstacles_indices(m), N, M, K)
dt_array = np.array(dt)
dt_filtered = dt_array[(dt_array != 0) & (dt_array != max_int)]
avg_dt = np.mean(dt_filtered)
print(avg_dt)