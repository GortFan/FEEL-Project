import numpy as np
from shape3d import cylinder_mask, cuboid_mask
import math
import pyvista as pv
import scipy.ndimage as sp
import random
from multiprocessing import Process, Queue

def fitness(m, c_count) -> float:
    penalty = 0
    if c_count < math.ceil(m.shape[0]*m.shape[1]*m.shape[2]*0.35):
        penalty = (m.shape[0]*m.shape[1]*m.shape[2] - c_count) + (m.shape[0]*m.shape[1]*m.shape[2] // 100)
    return (eval1(m) + eval2(m) + penalty)

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

def initialize_pop(pop_size: int) -> list:
    bounds = np.array(struct)
    lows = bounds[:, 0]
    highs = bounds[:, 1] + 1
    
    pop = np.random.randint(low=lows[:, None], high=highs[:, None], size=(len(struct), pop_size)).T
    return pop

def tournament_selection(generation, tournament_size=3):
    selected = random.sample(generation, tournament_size)
    selected_fitnesses = [ind[1] for ind in selected]
    winner_idx = np.argmin(selected_fitnesses)
    return np.array(selected[winner_idx][0])
    
def retain_elites(generation, elite_size=1):
    sorted_gen = sorted(generation, key=lambda x: x[1])
    return [np.array(ind[0]) for ind in sorted_gen[:elite_size]]

def retain_worst(generation, worst_size=2):
    sorted_gen = sorted(generation, key=lambda x: x[1], reverse=True)
    return [np.array(ind[0]) for ind in sorted_gen[:worst_size]]

def crossover(parent1, parent2):
    mask = np.random.randint(0, 2, size=parent1.shape)
    return np.where(mask, parent1, parent2)

def mutation(chrom):
    idx = random.randint(0, len(chrom) - 1)
    min_val, max_val = struct[idx]
    chrom[idx] = random.randint(min_val, max_val)
    return chrom

def compute_diversity(pop: np.ndarray) -> float:
    diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    upper_tri = np.triu_indices_from(dists, k=1)
    mean_distance = dists[upper_tri].mean()
    return mean_distance

def sharing_function(distances: np.ndarray, sigma: float, alpha: float) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = distances < sigma
        shared = np.zeros_like(distances)
        shared[mask] = 1 - (distances[mask] / sigma) ** alpha
        return shared
    
def penalize_drift_sharing(pop: np.ndarray, fitnesses: np.ndarray, sigma: float, alpha: float) -> np.ndarray:
    diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    
    sh_matrix = sharing_function(dists, sigma, alpha)
    
    niche_counts = sh_matrix.sum(axis=1)
    
    niche_counts[niche_counts == 0] = 1
    
    adjusted_fitnesses = fitnesses * niche_counts
    return adjusted_fitnesses

def is_in_population(chromosome, population):
    return np.any(np.all(population == chromosome, axis=1))
    
def create_ridge(m_shape, individual_batch, process_index, queue):
    result = [None]*len(individual_batch)
    for idx, individual in individual_batch:
        mc = np.ones(m_shape)
        r = cuboid_mask(matrix=mc,
                        base_z=0,
                        base_y=125,
                        base_x=125,
                        cuboid_depth=individual[0],
                        cuboid_height=individual[1],
                        cuboid_width=individual[2],
                        yaw=individual[3],
                        pitch=individual[4],
                        roll=individual[5],
                        taper_width=individual[6],
                        taper_height=individual[7])
        mc[r] = 0
        c = np.sum(mc == 1)
        f = fitness(mc, c)
        result[idx] = f
    queue.put((process_index,result))
        
def parallelize_ridge_evaluation(num_processes: int, fitnesses, pop, pop_size, m_shape):
    queue = Queue()
    processes = []
    process_chunk_size = pop_size // num_processes
    for process_index in range(num_processes):
        start = process_index * process_chunk_size
        if process_index == num_processes - 1:
            end = pop_size
        else:
            end = start + process_chunk_size
        p = Process(target=create_ridge, args=(m_shape, pop[start:end], process_index, queue))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    for _ in range(num_processes):
        process_idx, result = queue.get()
        start = process_idx * process_chunk_size
        if process_index == num_processes - 1:
            end = pop_size
        else:
            end = start + process_chunk_size
        fitnesses[start:end] = result
        
    
def genetic_algorithm(m, generation_qty, pop_size, elite_size, worst_size, mutation_rate, diversity_threshold, diversity_decay_rate, sigma, alpha):
    m[0, :, :] = 0
    children_qty = pop_size - elite_size - worst_size
    mutation_qty = math.ceil(mutation_rate*children_qty)
    
    pop = initialize_pop(pop_size)
    
    best_raw_fitness = float('inf')
    best_chrom = None
    fitness_history = []
    # The loop contents must be serialized because the order of chained operations must be consistent
    for generation_number in range(generation_qty):
        
        fitnesses = np.zeros(pop_size)

        # The nested loop contents are independent of each other so this can be parallelized
        for i in range(pop_size):
            print(i)
            mc = m.copy()
            r = cuboid_mask(matrix=mc,
                            base_z=0,
                            base_y=125,
                            base_x=125,
                            cuboid_depth=pop[i][0],
                            cuboid_height=pop[i][1],
                            cuboid_width=pop[i][2],
                            yaw=pop[i][3],
                            pitch=pop[i][4],
                            roll=pop[i][5],
                            taper_width=pop[i][6],
                            taper_height=pop[i][7])
            mc[r] = 0
            c = np.sum(mc == 1)
            fitnesses[i] = fitness(mc, c)   
        
        current_min_idx = np.argmin(fitnesses)
        current_min_fitness = fitnesses[current_min_idx]
        print(current_min_fitness)
        
        if current_min_fitness < best_raw_fitness:
            best_raw_fitness = current_min_fitness
            best_chrom = pop[current_min_idx].copy()
        
        diversity = compute_diversity(pop)
        
        if diversity < diversity_threshold * np.exp(-diversity_decay_rate * generation_number) and generation_number != generation_qty - 1:
            print("diversity")
            fitnesses = penalize_drift_sharing(pop, fitnesses, sigma, alpha)    
        else:
            print("raw fitness is carried over")    

        generation = list(zip(pop.tolist(), fitnesses.tolist()))
        
        children = []
        for _ in range(children_qty):
            parent1 = tournament_selection(generation)
            parent2 = tournament_selection(generation)
            child = crossover(np.array(parent1), np.array(parent2))
            children.append(child)
        children = np.array(children)

        for _ in range(mutation_qty):
            idx = random.randint(0, children_qty - 1)
            children[idx] = mutation(children[idx])

        elites = np.array(retain_elites(generation, elite_size))
        worsts = np.array(retain_worst(generation, worst_size))

        pop = np.vstack([children, elites, worsts])
        
        if not is_in_population(best_chrom, pop):
            pop[children_qty] = best_chrom
        
        fitness_history.append(best_raw_fitness)
        

    min_fitness_idx = np.argmin(fitnesses)
    min_fitness = fitnesses[min_fitness_idx]
    best_chromosome = pop[min_fitness_idx].tolist()
    return min_fitness, best_chromosome, list(zip(pop.tolist(), fitnesses.tolist())), fitness_history

if __name__ == "__main__":
    m = np.ones((121,250,250))
    m[0, :, :] = 0
    
    generations = 100
    elite_size = 1
    worst_size = 1
    pop_size = 100
    mutation_rate = 0.1
    diversity_threshold = 5000
    threshold_decay_rate = 0.1
    sigma = 5
    alpha = 1
    print("hi")
    fmin, fmin_chrom, tgen, fitnesses = genetic_algorithm(
        m,
        generations,
        pop_size,
        elite_size,
        worst_size,
        mutation_rate,
        diversity_threshold,
        threshold_decay_rate,
        sigma,
        alpha
    )
    print(fmin)
    print(tgen)
    print(fmin_chrom)
    print(len(fitnesses))
    r = cuboid_mask(matrix=m,
                    base_z=0,
                    base_y=125,
                    base_x=125,
                    cuboid_depth=fmin_chrom[0],
                    cuboid_height=fmin_chrom[1],
                    cuboid_width=fmin_chrom[2],
                    yaw=fmin_chrom[3],
                    pitch=fmin_chrom[4],
                    roll=fmin_chrom[5],
                    taper_width=fmin_chrom[6],
                    taper_height=fmin_chrom[7])
    mc = m.copy()
    mc[r] = 0
    # Create a uniform grid
    grid = pv.ImageData(dimensions=np.array(mc.shape) + 1)
    grid.cell_data["values"] = mc.ravel(order="F")  # For correct orientation

    # Threshold to keep only m == 0
    thresholded = grid.threshold(0.1, invert=True)  # Keep values <= 0.1

    # Plot with lighting
    plotter = pv.Plotter()
    plotter.enable_shadows()
    plotter.add_mesh(thresholded, color="red", show_edges=False,
                     ambient=0.3, diffuse=0.7, specular=0.5)
    plotter.show()