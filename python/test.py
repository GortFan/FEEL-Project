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
    return edt.get()

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
        
    o_src = np.ones([1, m.shape[1], m.shape[2]]) 
    m2 = np.append(m, o_src, axis=0) 
    obstacles = get_obstacles_indices(m2)
    N, M, K = m2.shape
    
    obstacles = get_obstacles_indices(m)
    
    distance_transform = fc.dialsDijkstra3D_Implicit(generate_sources(m2), obstacles, N, M, K)
    return distance_transform

def fitness(m) -> float: 
    """Returns:
    Fitness, e1, e2, penalty, mssp, edt"""
    mssp = eval2(m)
    mssp = mssp[:-m.shape[1]*m.shape[2]]
    mssp_mat = np.array(mssp).reshape(m.shape[0], m.shape[1], m.shape[2])
    dead_c = np.where(mssp_mat == 2147483647)
    edt = eval1(m)
    edt[dead_c] = 0
    
    # e1
    edt_nz = edt[edt != 0]
    if len(edt_nz) == 0:
        e1 = np.iinfo(np.uint32).max
    else:
        e1 = np.mean(edt_nz)
        
    # e2
    obstacle_id = 2147483647-1
    filtered_distance_transform = [d for d in mssp if 0 < d < obstacle_id]
    if len(filtered_distance_transform) == 0:
        e2 = np.iinfo(np.uint32).max
    else:
        e2 = np.mean(filtered_distance_transform) / 10
    
    # p
    penalty = 0
    usable_c = m.shape[0]*m.shape[1]*m.shape[2] - len(np.where(edt == 0)[0])
    if usable_c < math.ceil(m.shape[0]*m.shape[1]*m.shape[2]*0.35):
        penalty = (m.shape[0]*m.shape[1]*m.shape[2] - usable_c) + (m.shape[0]*m.shape[1]*m.shape[2] // 100)
    f = e1 + e2 + penalty
    return f, e1, e2, penalty, mssp_mat, edt, usable_c

class TestDistanceCalc(unittest.TestCase):
    def test_edt_formation_orientation_horizontal_block(self):
        m = np.ones((4,2,2))
        m[0, :, :] = 0
        m[2, :, :] = 0
        f, e1, e2, p, mssp, edt, c = fitness(m)
        self.assertTrue(np.array_equal(m[0, :, :], edt[0, :, :]))
        # check if this subset which is normally a dist has been set to 0 due to obstruction (dead catalyst)
        self.assertTrue(np.all(edt[1, :, :] == 0)) 
        
    def test_mssp_formation_orientation_horizontal_block(self):
        m = np.ones((4,2,2))
        m[0, :, :] = 0
        m[2, :, :] = 0
        f, e1, e2, p, mssp, edt, c = fitness(m)
        self.assertTrue(np.all(mssp[0, :, :] == 2147483646)) # base, id
        self.assertTrue(np.all(mssp[2, :, :] == 2147483646)) # obstruction, same as base
        self.assertTrue(np.all(mssp[1, :, :] == 2147483647)) # dead catalyst, id
    
    def test_mssp_formation_orientation_diagonal_block(self):
        m = np.ones((4,2,2))
        m[0, :, :] = 0
        m[2, 1, :] = 0
        m[1, 0, :] = 0
        f, e1, e2, p, mssp, edt, c = fitness(m)
        self.assertTrue(np.all(mssp[0, :, :] == 2147483646)) # base, id
        self.assertTrue(np.all(mssp[2, 1, :] == 2147483646)) # obstruction, same as base
        self.assertTrue(np.all(mssp[1, 0, :] == 2147483646)) # obstruction, same as base
        self.assertTrue(np.all(mssp[1, 1, :] == 34)) # goes through diagonal with path int(10*(sqrt(2) + 1 + 1)) = 34
        
    def test_edt_formation_orientation_diagonal_block(self):
        m = np.ones((4,2,2))
        m[0, :, :] = 0
        m[2, 1, :] = 0
        m[1, 0, :] = 0
        f, e1, e2, p, mssp, edt, c = fitness(m)
        self.assertTrue(np.all(edt[0, :, :] == 0)) # base, id
        self.assertTrue(np.all(edt[2, 1, :] == 0)) # obstruction, same as base
        self.assertTrue(np.all(edt[1, 0, :] == 0)) # obstruction, same as base. note: made 0 in post processing due to it being dead catalyst
        self.assertTrue(np.all(edt[1, 1, :] == 1)) # blockages do not affect edt, included tests for comprehensiveness

    def test_fitness_horizontal_block(self):
        m = np.ones((4,2,2))
        m[0, :, :] = 0
        m[2, :, :] = 0
        f, e1, e2, p, mssp, edt, c = fitness(m)
        self.assertTrue(e1 == 1)
        self.assertTrue(e2 == 1)
        self.assertTrue(p == (m.shape[0]*m.shape[1]*m.shape[2] - c) + (m.shape[0]*m.shape[1]*m.shape[2] // 100))
        self.assertTrue(f > e1 + e2)
        self.assertTrue(c == 4)
        self.assertTrue(f == 1 + 1 + 12)
        
    def test_c_count_horizontal_block(self):
        m = np.ones((4,2,2))
        m[0, :, :] = 0
        m[2, :, :] = 0
        f, e1, e2, p, mssp, edt, c = fitness(m)
        self.assertTrue(c == 4)
    
    def test_c_count_diagonal_block(self):
        m = np.ones((4,2,2))
        m[0, :, :] = 0
        m[2, 1, :] = 0
        m[1, 0, :] = 0
        f, e1, e2, p, mssp, edt, c = fitness(m)
        self.assertTrue(c == 8)
    
    def test_handling_full_space(self):
        m = np.ones((4,2,2))
        m[:, :, :] = 0
        f, e1, e2, p, mssp, edt, c = fitness(m)
        self.assertTrue(e1 == np.iinfo(np.uint32).max)
        self.assertTrue(e2 == np.iinfo(np.uint32).max)
        self.assertTrue(p == m.shape[0]*m.shape[1]*m.shape[2]) # c_usable is 0 and volume < 100 so int div is 0. hi future me, in case u look at this im right so dw abt it
    
    def test_res(self):
        param = [40, 132, 146, 44, 51, 4, 6]
        m = np.ones((61, 124, 124))
        m[0, :, :] = 0
        r = elliptical_cylinder_mask(
            matrix=m,
            base_z=0,
            base_y=m.shape[1] // 2,
            base_x=m.shape[2] // 2,
            cylinder_length=param[0],
            radius_y=param[1],
            radius_x=param[2],
            yaw=param[3],
            pitch=param[4],
            roll=0,
            taper_z=param[5] / 10
        )     
        c = cuboid_mask(
            matrix=m,
            base_z=0,
            base_y=m.shape[1] // 2,
            base_x=m.shape[2] // 2,
            cuboid_depth=param[0],
            cuboid_height=param[1],
            cuboid_width=param[2],
            yaw=param[3],
            pitch=param[4],
            roll=0,
            taper_width=param[5] / 10,
            taper_height=param[6] / 10
        )
        m[c] = 0
        c_req = (m.shape[0] - 1)*m.shape[1]*m.shape[2]
        c_curr = np.count_nonzero(m == 1)
        c_layer = m.shape[1]*m.shape[2]
        num_layer = math.ceil((c_req - c_curr) / c_layer)
        m_appended_c = np.append(m, np.ones((num_layer, m.shape[1], m.shape[2])), axis=0)
        f, e1, e2, p, _, _, c = fitness(m_appended_c)
        self.assertTrue(c_req <= c)
        self.assertTrue(m.shape[0] <= m_appended_c.shape[0])
if __name__ == '__main__':
    unittest.main()