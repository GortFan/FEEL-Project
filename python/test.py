import sys, os
project_root = os.path.dirname(os.path.dirname(__file__))

sys.path.append(os.path.join(project_root, 'build', 'Release'))

import unittest, math
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as cp_ndimage
import FEELcpp as fc
from shape3d import elliptical_cylinder_mask, cuboid_mask

def eval1(m) -> float:
    """Returns two values the 1. edt value avg, 2. edt matrix"""
    m = cp.asarray(m)
    edt = cp_ndimage.distance_transform_edt(m)
    edt_nz = edt[edt != 0]
    if len(edt_nz) == 0:
        return np.iinfo(np.uint32).max, edt
    m_edt = cp.mean(edt_nz)
    return float(m_edt.get()), edt.get()

def eval2(m):
    def get_obstacles_indices(m):
        obstacle_positions_3D = np.where(m == 0)
        _, M, K = m.shape
        obstacle_positions_1D = (obstacle_positions_3D[0] * M * K +
                                obstacle_positions_3D[1] * K +
                                obstacle_positions_3D[2])
        return obstacle_positions_1D.tolist()

    def generate_sources(m):
        N, M, K = m.shape
        
        last_layer = m[N-1, :, :]
        source_positions_2d = np.where(last_layer == 1)
        
        y_coords = source_positions_2d[0]
        z_coords = source_positions_2d[1]
        x_coords = np.full_like(y_coords, N-1)
        
        sources = (x_coords * M * K + y_coords * K + z_coords).tolist()
        return sources 
        
    obstacle_id = 2147483647-1
    o_src = np.ones([1, m.shape[1], m.shape[2]]) 
    m2 = np.append(m, o_src, axis=0) 
    obstacles = get_obstacles_indices(m2)
    N, M, K = m2.shape
    
    obstacles = get_obstacles_indices(m)
    
    distance_transform = fc.dialsDijkstra3D_Implicit(generate_sources(m2), obstacles, N, M, K)
    blocked_voids_count = distance_transform.count(2147483647)
    
    filtered_distance_transform = [d for d in distance_transform if 0 < d < obstacle_id]
    if len(filtered_distance_transform) == 0:
        return np.iinfo(np.uint32).max, np.array(distance_transform).reshape((N, M, K)), blocked_voids_count
    return np.mean(filtered_distance_transform) / 10, np.array(distance_transform).reshape((N, M, K)), blocked_voids_count

def fitness(m, c_count) -> float: 
    penalty = 0
    e1, _ = eval1(m)
    e2, _, dead_c = eval2(m)
    usable_c = c_count  - dead_c
    if usable_c < math.ceil(m.shape[0]*m.shape[1]*m.shape[2]*0.35):
        penalty = (m.shape[0]*m.shape[1]*m.shape[2] - usable_c) + (m.shape[0]*m.shape[1]*m.shape[2] // 100)
    f = e1 + e2 + penalty
    return f, e1, e2, penalty

class TestDistanceCalc(unittest.TestCase):

    def test_fitness_if_filled_matrix(self):
        m = np.ones((5, 5, 5))
        m[0, :, :] = 0
        r = cuboid_mask(matrix=m,
                    base_z=0,
                    base_y=m.shape[1] // 2,
                    base_x=m.shape[2] // 2,
                    cuboid_depth=5,
                    cuboid_height=5,
                    cuboid_width=5,
                    yaw=0,
                    pitch=0,
                    roll=0,
                    taper_height=0 / 10,
                    taper_width=0 / 10)
        m[r] = 0
        c = np.sum(m == 1)
        f, e1, e2, p = fitness(m, c)
        p = (m.shape[0]*m.shape[1]*m.shape[2] - c) + (m.shape[0]*m.shape[1]*m.shape[2] // 100)
        self.assertEqual(f, np.iinfo(np.uint32).max*2 + p)
    
    def test_fitness_if_empty_matrix(self):
        m = np.ones((5,5,5))
        m[0, :, :] = 0
        r = cuboid_mask(matrix=m,
                    base_z=2,
                    base_y=m.shape[1] // 2,
                    base_x=m.shape[2] // 2,
                    cuboid_depth=0,
                    cuboid_height=0,
                    cuboid_width=0,
                    yaw=0,
                    pitch=0,
                    roll=0,
                    taper_height=0 / 10,
                    taper_width=0 / 10)
        m[r] = 0
        c = np.sum(m == 1)
        f, _, _, _ = fitness(m, c)
        self.assertEqual(f, (m.shape[0] / 2) * 2)
    
    def test_fitness_with_horizontal_blockage(self):
        m = np.ones((5,5,5))
        m[0, :, :] = 0
        r = cuboid_mask(matrix=m,
                    base_z=3,
                    base_y=m.shape[1] // 2,
                    base_x=m.shape[2] // 2,
                    cuboid_depth=1,
                    cuboid_height=5,
                    cuboid_width=5,
                    yaw=0,
                    pitch=0,
                    roll=0,
                    taper_height=0 / 10,
                    taper_width=0 / 10)
        m[r] = 0
        c = np.sum(m == 1)
        f, e1, e2, p = fitness(m, c)
        self.assertEqual(p, 101) #not 101.25 b/c int division 
        self.assertEqual(e1, 1)
        self.assertEqual(e2, 1)
        self.assertEqual(f, 1 + 1 + 101)
    
    def test_fitness_with_vertical_blockage(self):
        m = np.ones((5,5,5))
        m[0, :, :] = 0
        r = cuboid_mask(matrix=m,
                    base_z=0,
                    base_y=m.shape[1] // 2,
                    base_x=m.shape[2] // 2,
                    cuboid_depth=1,
                    cuboid_height=5,
                    cuboid_width=8,
                    yaw=0,
                    pitch=90,
                    roll=0,
                    taper_height=0 / 10,
                    taper_width=0 / 10)
        m[r] = 0
        msspavg, _, _ = eval2(m)
        self.assertEqual(msspavg, np.average(a=[4,3,2,1]))
        
    def test_location_of_obstacles(self):
        m = np.ones((10, 10, 10))
        m[0, :, :] = 0
        r = cuboid_mask(matrix=m,
                    base_z=2,
                    base_y=m.shape[1] // 2,
                    base_x=m.shape[2] // 2,
                    cuboid_depth=1,
                    cuboid_height=10,
                    cuboid_width=10,
                    yaw=0,
                    pitch=0,
                    roll=0,
                    taper_height=0 / 10,
                    taper_width=0 / 10)
        m[r] = 0
        self.assertTrue(100 == np.sum(m[3, :, :]) and 0 == np.sum(m[2, :, :]))
    
    def test_value_of_blocked_voids(self):
        m = np.ones((10, 10, 10))
        m[0, :, :] = 0
        r = cuboid_mask(matrix=m,
                    base_z=2,
                    base_y=m.shape[1] // 2,
                    base_x=m.shape[2] // 2,
                    cuboid_depth=1,
                    cuboid_height=10,
                    cuboid_width=10,
                    yaw=0,
                    pitch=0,
                    roll=0,
                    taper_height=0 / 10,
                    taper_width=0 / 10)
        m[r] = 0
        _, ddt, _ = eval2(m)
        self.assertEqual(2147483647*m.shape[1]*m.shape[2], np.sum(ddt[1, :, :]))
    
    def test_value_of_obstacles(self):
        m = np.ones((10, 10, 10))
        m[0, :, :] = 0
        r = cuboid_mask(matrix=m,
                    base_z=2,
                    base_y=m.shape[1] // 2,
                    base_x=m.shape[2] // 2,
                    cuboid_depth=1,
                    cuboid_height=10,
                    cuboid_width=10,
                    yaw=0,
                    pitch=0,
                    roll=0,
                    taper_height=0 / 10,
                    taper_width=0 / 10)
        m[r] = 0
        _, ddt, _ = eval2(m)
        self.assertEqual(2147483646*m.shape[1]*m.shape[2], np.sum(ddt[2, :, :]))
    
if __name__ == '__main__':
    unittest.main()