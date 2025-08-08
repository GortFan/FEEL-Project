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

def initialize_random() -> list:
    bounds = np.array(struct)
    lows = bounds[:, 0]
    highs = bounds[:, 1] + 1
    pop = np.random.randint(low=lows, high=highs)
    pop = pop.tolist()
    return pop

def eval1(m) -> float:
    m = m.copy()
    edt = sp.distance_transform_edt(m)
    edt_nz = edt[edt != 0]
    m_edt = np.mean(edt_nz)
    return m_edt

def eval2(m) -> float:
    m = m.copy()
    m[m == 0] = np.nan 
    
    o_src = np.zeros([1, m.shape[1]]) 
    m2 = np.append(m, o_src, axis=0) 
    
    nans = np.isnan(m2)
    
    edt2 = sp.distance_transform_edt(m2)
    
    edt2[nans] = 0
    
    edt2_nz = edt2[edt2 != 0]
    m_edt2 = np.mean(edt2_nz)
    return m_edt2

def fitness(m, c_count, alpha) -> float:
    penalty = 0
    threshold = math.ceil(m.shape[0]*m.shape[1]*0.35)
    if c_count < threshold:
        penalty = alpha * max(0, threshold - c_count)
    return (eval1(m) + eval2(m) + penalty)

def initialize_man(signature, fitness):
    return (signature, fitness)

def perturb(solution_signature):
    new_signature = solution_signature.copy()
    i = random.randint(0, len(solution_signature)-1)
    low, high = struct[i]
    
    delta = random.choice([-1,1])
    new_val = new_signature[i] + delta
    

    new_signature[i] = max(low, min(high, new_val))
    return new_signature
    
def SimulatedAnnealing(m, T_max, T_min, E_th, alpha):

    solution = initialize_man([136, 95, 10, 10], 36.39819591445758)

    mc = m.copy()
    r = rectangle_mask(m, 0, 50, solution[0][0], solution[0][1], 0, solution[0][2] // 10, solution[0][3] // 10)
    mc[r] = 0
    c = np.sum(mc == 1)
    f = fitness(mc, c, alpha)
    solution = (solution[0], f)
    T = T_max

    while T > T_min and solution[1] > E_th:
        new_signature = perturb(solution_signature=solution[0])
        mc = m.copy()
        r = rectangle_mask(m, 0, 50, new_signature[0], new_signature[1], 0, new_signature[2] // 10, new_signature[3] // 10)
        mc[r] = 0
        c = np.sum(mc == 1)
        f = fitness(mc, c, alpha)
        E_delta = f - solution[1]
        print(f)
        if E_delta < 0 or random.random() < np.exp(-E_delta / T):
            solution = (new_signature, f)
            print("NEW FITNESS:", f)
        T *= alpha
    return solution

if __name__ == "__main__":
    import collections

    m = np.ones((101, 100))
    m[0, :] = 0
    
    results_counter = collections.Counter()
    
    for _ in range(100):
        solution = SimulatedAnnealing(m, 100, 1e-6, 30, 0.99)
        key = (tuple(solution[0]), solution[1])
        results_counter[key] += 1

    for sol, count in results_counter.most_common():
        print(f"Solution {sol} found {count} times")
