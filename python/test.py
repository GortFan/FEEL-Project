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
    m = cp.asarray(m)
    edt = cp_ndimage.distance_transform_edt(m)
    edt_nz = edt[edt != 0]
    if len(edt_nz) == 0:
        return np.iinfo(np.uint32).max
    m_edt = cp.mean(edt_nz)
    return float(m_edt.get())

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
    filtered_distance_transform = [d for d in distance_transform if 0 < d < obstacle_id]
    if len(filtered_distance_transform) == 0:
        return np.iinfo(np.uint32).max, np.array(distance_transform).reshape((N, M, K))
    return np.mean(filtered_distance_transform) / 10, np.array(distance_transform).reshape((N, M, K))

def fitness(m, c_count) -> float: 
    penalty = 0
    e1 = eval1(m)
    e2, edt2 = eval2(m)
    if c_count < math.ceil(m.shape[0]*m.shape[1]*m.shape[2]*0.35):
        penalty = (m.shape[0]*m.shape[1]*m.shape[2] - c_count) + (m.shape[0]*m.shape[1]*m.shape[2] // 100)
    f = e1 + e2 + penalty
    return f, e1, e2, penalty

def make_cuboid(m, base_x, base_y, base_z, cuboid_depth, cuboid_height, cuboid_width, yaw=0, pitch=0, roll=0, taper_height=0, taper_width=0):
    r = cuboid_mask(
        matrix=m,
        base_x=base_x,
        base_y=base_y,
        base_z=base_z,
        cuboid_depth=cuboid_depth,
        cuboid_height=cuboid_height,
        cuboid_width=cuboid_width,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        taper_height=taper_height,
        taper_width=taper_width
    )
    m[r] = 0
    c = np.sum(m == 1)
    return m, c

def make_cyl(m, base_x, base_y, base_z, cylinder_length, radius_x, radius_y, yaw, pitch, roll, taper_z):
    m[0, :, :] = 0
    r = elliptical_cylinder_mask(
        matrix=m,
        base_x=base_x,
        base_y=base_y,
        base_z=base_z,
        cylinder_length=cylinder_length,
        radius_x=radius_x,
        radius_y=radius_y,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        taper_z=taper_z
    )
    m[r] = 0
    c = np.sum(m == 1)
    return m, c

class TestDistanceCalc(unittest.TestCase):

    def test_fitness_if_filled_matrix(self):
        m = np.ones((10, 10, 10))
        m[0, :, :] = 0
        mc, c = make_cuboid(
            m=m,
            base_x=5,
            base_y=5,
            base_z=0,
            cuboid_depth=10,
            cuboid_height=10,
            cuboid_width=10,
            yaw=0,
            pitch=0,
            roll=0,
            taper_height=0,
            taper_width=0
        )
        f, _, _, _ = fitness(mc, c)
        p = (m.shape[0]*m.shape[1]*m.shape[2] - c) + (m.shape[0]*m.shape[1]*m.shape[2] // 100)
        self.assertEqual(f, np.iinfo(np.uint32).max*2 + p)
    
    def test_fitness_if_empty_matrix(self):
        m = np.ones((13, 10, 10))
        m[0, :, :] = 0
        mc, c = make_cuboid(
            m=m,
            base_x=5,
            base_y=5,
            base_z=0,
            cuboid_depth=0,
            cuboid_height=0,
            cuboid_width=0,
            yaw=0,
            pitch=0,
            roll=0,
            taper_height=0,
            taper_width=0
        )
        f, e1, e2, p = fitness(mc, c)
        print(f, e1, e2, p)
        self.assertEqual(f, (m.shape[0] / 2) * 2)
    
    def test_fitness_with_no_blockage(self):
        pass
    
    def test_fitness_with_horizontal_blockage(self):
        pass
    
    def test_fitness_with_2thick_diagonal_blockage(self):
        pass
    
    def test_fitness_with_1thick_diagonal_blockage(self):
        pass
    
    def test_fitness_with_vertical_blockage(self):
        pass
    
    def test_location_of_obstacles(self):
        m = np.ones((10, 10, 10))
        m[0, :, :] = 0
        mc, _ = make_cuboid(
            m=m,
            base_x=5,
            base_y=5,
            base_z=2,
            cuboid_depth=1,
            cuboid_height=10,
            cuboid_width=10,
            yaw=0,
            pitch=0,
            roll=0,
            taper_height=0,
            taper_width=0
        )
        self.assertTrue(100 == np.sum(mc[3, :, :]) and 0 == np.sum(mc[2, :, :]))
    
    def test_value_of_blocked_voids(self):
        m = np.ones((10, 10, 10))
        m[0, :, :] = 0
        mc, _ = make_cuboid(
            m=m,
            base_x=5,
            base_y=5,
            base_z=2,
            cuboid_depth=1,
            cuboid_height=10,
            cuboid_width=10,
            yaw=0,
            pitch=0,
            roll=0,
            taper_height=0,
            taper_width=0
        )
        _, ddt = eval2(mc)
        self.assertEqual(2147483647*m.shape[1]*m.shape[2], np.sum(ddt[1, :, :]))
    
    def test_value_of_obstacles(self):
        m = np.ones((10, 10, 10))
        m[0, :, :] = 0
        mc, _ = make_cuboid(
            m=m,
            base_x=5,
            base_y=5,
            base_z=2,
            cuboid_depth=1,
            cuboid_height=10,
            cuboid_width=10,
            yaw=0,
            pitch=0,
            roll=0,
            taper_height=0,
            taper_width=0
        )
        _, ddt = eval2(mc)
        self.assertEqual(2147483646*m.shape[1]*m.shape[2], np.sum(ddt[2, :, :]))
if __name__ == '__main__':
    unittest.main()