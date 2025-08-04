import time
import numpy as np
from shape3d import cylinder_mask, cuboid_mask
import math
import pyvista as pv
import scipy.ndimage as sp
import random
from multiprocessing import Process, Queue
import cupy as cp
import cupyx.scipy.ndimage as cp_ndimage

def eval1_gpu(m) -> float:
    m = cp.asarray(m)
    edt = cp_ndimage.distance_transform_edt(m)
    edt_nz = edt[edt != 0]
    m_edt = cp.mean(edt_nz)
    return float(m_edt.get())  # Convert back to CPU

def eval2_gpu(m) -> float:
    m_gpu = cp.asarray(m)
    m_gpu = m_gpu.copy()
    m_gpu[m_gpu == 0] = cp.nan  # mark ridge
    
    # append oxygen layer
    o_src = cp.zeros([1, m_gpu.shape[1], m_gpu.shape[2]]) 
    m2 = cp.append(m_gpu, o_src, axis=0) 
    
    # create mask with ridge against not ridge
    nans = cp.isnan(m2)
    
    edt2 = cp_ndimage.distance_transform_edt(m2)
    
    # remove ridge from matrix
    edt2[nans] = 0
    
    edt2_nz = edt2[edt2 != 0]
    m_edt2 = cp.mean(edt2_nz)
    return float(m_edt2.get())  # Convert back to CPU

def fitness_gpu(m, c_count) -> float: 
    penalty = 0  # No penalty because we are testing impacts w/o domain specific constraints
    if c_count < math.ceil(m.shape[0]*m.shape[1]*m.shape[2]*0.35):
        penalty = (m.shape[0]*m.shape[1]*m.shape[2] - c_count) + (m.shape[0]*m.shape[1]*m.shape[2] // 100)
    return (eval1_gpu(m) + eval2_gpu(m) + penalty)
            
def eval1(m) -> float:
    m = m.copy()
    edt = sp.distance_transform_edt(m)
    edt_nz = edt[edt != 0]
    m_edt = np.mean(edt_nz)
    return m_edt

def eval2(m) -> float:
    m = m.copy()
    m[m == 0] = np.nan #mark ridge
    
    #append oxygen layer
    o_src = np.zeros([1, m.shape[1], m.shape[2]]) 
    m2 = np.append(m, o_src, axis=0) 
    
    #create mask with ridge against not ridge
    nans = np.isnan(m2)
    
    edt2 = sp.distance_transform_edt(m2)
    
    #remove ridge from matrix
    edt2[nans] = 0
    
    edt2_nz = edt2[edt2 != 0]
    m_edt2 = np.mean(edt2_nz)
    return m_edt2

def fitness(m) -> float: 
    penalty = 0 # No penalty because we are testing impacts w/o domain specific constraints
    return (eval1(m) + eval2(m) + penalty)

# Rules:
# 1. Base_Z must be the height parameter divided by -2
# 2. Yaw cannot exceed 180 (this is overkill but analytically the shape will cease to exist at this value or before it due to integer math related stuff) 
# 2.a also 180 is for when base_z is 0 but this isnt the case here so just set Yaw limit to 90 since thats accounting for the lack of dividing by 2.

import pandas as pd
import numpy as np
from shape3d import cuboid_mask, cylinder_mask

# stage 1:
# df should contain everything for a single shape
# generation should be coarse but comprehensive
    # step does not have to be +1 but can be a integer rounded % of the range
    # for a single shape, vary 1 parameter by step and keep the rest fixed to get an idea of degree of influence of each param

# df for cuboid
columns = ['depth', 'height', 'width', 'yaw', 'pitch', 'roll', 'taper_width', 'taper_height', 'chromosome', 'fitness']
df = pd.DataFrame(columns=columns)

struct = [
    (1, 120),
    (1, 125),
    (1, 125),
    (1, 180),
    (1, 180),
    (1, 180),
    (0, 10),
    (0, 10),
]

def generate_single_allele_series(i):
    import pandas as pd
    columns = ['depth', 'height', 'width', 'yaw', 'pitch', 'roll', 'taper_width', 'taper_height', 'chromosome', 'fitness']
    df = pd.DataFrame(columns=columns)

    step = (struct[i][1] - struct[i][0]) * 0.01
    num_steps = int(1 / 0.01)
    allele_names = ['depth', 'height', 'width', 'yaw', 'pitch', 'roll', 'taper_width', 'taper_height']

    # Set baseline values (midpoint or a fixed value)
    baseline = [60, 125, 125, 0, 0, 0, 0, 0]
    for step_idx in range(num_steps):
        chrom = baseline.copy()
        chrom[i] = struct[i][0] + step * step_idx

        m = np.ones((121, 250, 250))
        m[0, :, :] = 0

        # Unpack chromosome for cuboid_mask
        c = cuboid_mask(
            matrix=m,
            base_z=0,
            base_y=m.shape[1] // 2,
            base_x=m.shape[2] // 2,
            cuboid_depth=chrom[0],
            cuboid_height=chrom[1],
            cuboid_width=chrom[2],
            yaw=chrom[3],
            pitch=chrom[4],
            roll=chrom[5],
            taper_width=chrom[6],
            taper_height=chrom[7],
        )
        m[c] = 0
        c = np.sum(m == 1)
        f = fitness_gpu(m,c)
        entry = dict(zip(allele_names, chrom))
        entry['chromosome'] = chrom
        entry['fitness'] = f
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)

    df.to_csv(f'1_{allele_names[i].capitalize()}_Series.csv', index=False)

if __name__ == "__main__":
    for i in range(8):
        print(i)
        generate_single_allele_series(i)
    
