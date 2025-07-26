from shape2d import rectangle_mask
import numpy as np
import math
import scipy.ndimage as sp
import random 

struct = [
    (1,198),
    (1,98),
    (0,10),
    (0,10),
]

bounds = np.array(struct)

class Particle:
    def __init__(self):
        # Initialize position randomly as floats inside bounds
        lows = bounds[:, 0]
        highs = bounds[:, 1] + 1  # keep +1 for randint style upper bound
        self.position = np.random.uniform(low=lows, high=highs)
        self.velocity = np.zeros_like(self.position)
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
    
    def update_velocity(self, global_best_position, w=0.5, c1=1.5, c2=1.5):
        r1 = np.random.rand(len(self.position))
        r2 = np.random.rand(len(self.position))
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self):
        self.position += self.velocity
        # Clamp to bounds
        self.position = np.maximum(self.position, bounds[:, 0])
        self.position = np.minimum(self.position, bounds[:, 1])

    def get_int_position(self):
        return np.round(self.position).astype(int)

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
    o_src = np.zeros([1, m.shape[1]]) 
    m2 = np.append(m, o_src, axis=0) 
    
    #create mask with ridge against not ridge
    nans = np.isnan(m2)
    
    edt2 = sp.distance_transform_edt(m2)
    
    #remove ridge from matrix
    edt2[nans] = 0
    
    edt2_nz = edt2[edt2 != 0]
    m_edt2 = np.mean(edt2_nz)
    return m_edt2

def fitness(m, c_count) -> float:
    penalty = 0
    threshold = math.ceil(m.shape[0]*m.shape[1]*0.35)
    if c_count < threshold:
        penalty = max(0, threshold - c_count)
    return (eval1(m) + eval2(m) + penalty)

def PSO_int(swarm_size, iterations):
    m_orig = m.copy() 
    swarm = [Particle() for _ in range(swarm_size)]

    # Evaluate initial particles
    global_best_fitness = float('inf')
    global_best_position = None
    for iter in range(iterations):
        for particle in swarm:
            mc = m.copy()
            chrom = particle.get_int_position()
            r = rectangle_mask(m_orig.copy(), 0, 50, chrom[0], chrom[1], 0, chrom[2], chrom[3])
            mc[r] = 0
            c = np.sum(mc == 1)
            f = fitness(mc, c)
            if f < particle.best_fitness:
                particle.best_fitness = f
                particle.best_position = particle.position.copy()

            if f < global_best_fitness:
                global_best_fitness = f
                global_best_position = particle.position.copy()
                print(iter, global_best_fitness)
                
            particle.update_velocity(global_best_position)
            particle.update_position()
    return global_best_fitness, global_best_position

if __name__ == "__main__":
    m = np.ones((101, 100))
    m[0, :] = 0
    f, p = PSO_int(500, 100)
    print(f, p)


    