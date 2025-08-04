import sys, os, time, math, random
import numpy as np
from shape3d import cylinder_mask, cuboid_mask
import pyvista as pv
from multiprocessing import Process, Queue
import cupy as cp
import cupyx.scipy.ndimage as cp_ndimage
import pandas as pd

project_root = os.path.dirname(os.path.dirname(__file__))

sys.path.append(os.path.join(project_root, 'build', 'Release'))

import mybindings as bindings

def eval1(m) -> float:
    """
    Uses scipy EDT function with CUDA GPU acceleration.
    """
    m = cp.asarray(m)
    edt = cp_ndimage.distance_transform_edt(m)
    edt_nz = edt[edt != 0]
    m_edt = cp.mean(edt_nz)
    return float(m_edt.get()) 

def eval2(m, use_dials=False):
    """
    Uses custom written dails algorithm in C++ using purely the CPU. 
    Works for scenarios when obstructions would intercept EDT straight lines.
    """
    def get_obstacles_indices(m):
        """Returns a list of obstacles (0=obstacle, 1=traversible) in 1D array indexing style"""
        obstacle_positions_3D = np.where(m == 0)
        _, M, K = m.shape
        obstacle_positions_1D = (obstacle_positions_3D[0] * M * K +
                                obstacle_positions_3D[1] * K +
                                obstacle_positions_3D[2])
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
    if use_dials:
        N, M, K = m.shape
        CPP_INT_MAX = 2147483647 # from std::numeric_limits<int>::max(), for obstacle nodes

        adj = bindings.makeAdjMatrix3D(N, M, K, get_obstacles_indices(m))
        distance_transform = bindings.dialsDijkstra3D(adj, generate_sources(m), N, M, K)
        
        filtered_distance_transform = [d for d in distance_transform if 0 < d < CPP_INT_MAX]
        
        return np.mean(filtered_distance_transform)
    else:
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

def fitness(m, c_count) -> float: 
    """
    Encapsulates the eval logic and adds penalty based on catalyst preservation requirements (arbitrary magic number).
    """
    penalty = 0
    if c_count < math.ceil(m.shape[0]*m.shape[1]*m.shape[2]*0.35):
        print("penalty")
        penalty = (m.shape[0]*m.shape[1]*m.shape[2] - c_count) + (m.shape[0]*m.shape[1]*m.shape[2] // 100)
    return (eval1(m) + eval2(m) + penalty)

struct = [
    (1, 59),
    (1, 62),
    (1, 62),
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
    for idx, individual in enumerate(individual_batch):
        mc = np.ones(m_shape)
        r = cuboid_mask(matrix=mc,
                        base_z=0,
                        base_y=mc.shape[1] // 2,
                        base_x=mc.shape[2] // 2,
                        cuboid_depth=individual[0],
                        cuboid_height=individual[1],
                        cuboid_width=individual[2],
                        yaw=individual[3],
                        pitch=individual[4],
                        roll=0,
                        taper_width=individual[5],
                        taper_height=individual[6])
        mc[r] = 0
        c = np.sum(mc == 1)
        f = fitness(mc, c)
        result[idx] = f
        print(f)
    print(result)
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
        if process_idx == num_processes - 1:
            end = pop_size
        else:
            end = start + process_chunk_size
        fitnesses[start:end] = result
    return fitnesses
    
def genetic_algorithm(m, generation_qty, pop_size, elite_size, worst_size, mutation_rate, diversity_threshold, diversity_decay_rate, sigma, alpha):
    children_qty = pop_size - elite_size - worst_size
    mutation_qty = math.ceil(mutation_rate*children_qty)
    
    pop = initialize_pop(pop_size)
    
    best_raw_fitness = float('inf')
    best_chrom = None
    fitness_history = []
    generation_data = []
    
    for generation_number in range(generation_qty):
        # Get raw fitnesses (never modify these!)
        raw_fitnesses = parallelize_ridge_evaluation(5, np.zeros(pop_size), pop, pop_size, m.shape)
        current_min_idx = np.argmin(raw_fitnesses)
        current_min_fitness = raw_fitnesses[current_min_idx]
        
        # Update global best using RAW fitnesses
        if current_min_fitness < best_raw_fitness:
            best_raw_fitness = current_min_fitness
            best_chrom = pop[current_min_idx].copy()
        
        diversity = compute_diversity(pop)
        
        generation_data.append({
            'generation': generation_number,
            'best_fitness': best_raw_fitness,
            'current_min_fitness': current_min_fitness,
            'best_chromosome': best_chrom,
            'avg_fitness': np.mean(raw_fitnesses),
            'std_fitness': np.std(raw_fitnesses),
            'diversity': diversity
        })
        
        # Create SEPARATE penalized fitnesses for selection
        current_diversity_threshold = diversity_threshold * np.exp(-diversity_decay_rate * generation_number)
        if diversity < current_diversity_threshold and generation_number != generation_qty - 1:
            print("diversity")
            selection_fitnesses = penalize_drift_sharing(pop, raw_fitnesses, sigma, alpha)
        else:
            print("raw fitness is carried over")
            selection_fitnesses = raw_fitnesses.copy()  # Use copy to be safe

        # Use penalized fitnesses for selection only
        generation = list(zip(pop.tolist(), selection_fitnesses.tolist()))
        
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
    
    # For final reporting, use raw fitnesses
    final_raw_fitnesses = parallelize_ridge_evaluation(10, np.zeros(pop_size), pop, pop_size, m.shape)
    
    df = pd.DataFrame(generation_data)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"Individual_Run_Fitness_History_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"Fitness history saved to {filename}")
    
    # Debug with raw fitnesses
    min_fitness_idx = np.argmin(final_raw_fitnesses)
    min_fitness = final_raw_fitnesses[min_fitness_idx]
    best_chromosome = pop[min_fitness_idx].tolist()
    
    print(f"Global best fitness: {best_raw_fitness}")
    print(f"Global best chromosome: {best_chrom}")
    print(f"Final generation best fitness: {min_fitness}")
    print(f"Final generation best chromosome: {best_chromosome}")
    
    # Return the correct global best
    return best_raw_fitness, best_chrom.tolist(), list(zip(pop.tolist(), final_raw_fitnesses.tolist())), fitness_history

def GA_dispatch():
    m = np.ones((61, 124, 124))
    m[0, :, :] = 0
    
    generations = 10
    elite_size = 1
    worst_size = 1
    pop_size = 20
    mutation_rate = 0.1
    diversity_threshold = 5000
    threshold_decay_rate = 0.1
    sigma = 30
    alpha = 1
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
    return fmin, fmin_chrom, tgen, fitnesses
    
def profiler():
    import time
    start = time.time()
    t0 = time.time()
    m = np.ones((61,124,124))
    t1 = time.time()
    m[0, :, :] = 0
    t2 = time.time()
    print(f"Time to create m: {t1 - t0:.4f} seconds")
    print(f"Time to zero m[0,:,:]: {t2 - t1:.4f} seconds")
    e = [47, 62, 62, 176, 178, 0, 0]
    t3 = time.time()
    r = cuboid_mask(matrix=m,
                    base_z=0,
                    base_y=62,
                    base_x=62,
                    cuboid_depth=e[0],
                    cuboid_height=e[1],
                    cuboid_width=e[2],
                    yaw=e[3],
                    pitch=e[4],
                    roll=0,
                    taper_width=e[5],
                    taper_height=e[6])
    t4 = time.time()
    print(f"Time for cuboid_mask: {t4 - t3:.4f} seconds")

    mc = m.copy()
    t5 = time.time()
    print(f"Time to copy m: {t5 - t4:.4f} seconds")

    mc[r] = 0
    t6 = time.time()
    print(f"Time to set mc[r]=0: {t6 - t5:.4f} seconds")

    c = np.sum(mc == 1)
    t7 = time.time()
    print(f"Time to sum mc==1: {t7 - t6:.4f} seconds")

    f = fitness(mc, c)
    t8 = time.time()
    print(f"Time for fitness: {t8 - t7:.4f} seconds")

    print(f"Fitness value: {f}")
    print(f"Total execution time: {t8 - start:.4f}")
    
    #copy paste this at the end whenever mc exists to output how it looks
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
    
def single_test():
    m = np.ones((61, 124, 124))
    m[0, :, :] = 0
    c2 = [47, 62, 28, 73, 178, 0, 0]
    r = cuboid_mask(matrix=m,
                    base_z=0,
                    base_y=m.shape[1] // 2,
                    base_x=m.shape[2] // 2,
                    cuboid_depth=c2[0],
                    cuboid_height=c2[1],
                    cuboid_width=c2[2],
                    yaw=c2[3],
                    pitch=c2[4],
                    roll=0,
                    taper_width=c2[5],
                    taper_height=c2[6])
    m[r] = 0
    c = np.sum(m == 1)
    f = fitness(m, c)
    print(f)

if __name__ == "__main__":
    num_runs = 1
    all_results = [None]*num_runs
    for i in range(num_runs):
        seed = random.randint(1, 1024)
        np.random.seed(seed)
        random.seed(seed)
        cp.random.seed(seed)
        result = GA_dispatch()
        
        all_results[i] = ({
            'run': i,
            'seed': seed,
            'best_fitness': result[0],
            'best_chromosome': result[1]
        })
        print(f"Run {i} completed")
    
    df = pd.DataFrame(all_results)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"ga_multiple_runs_{num_runs}_{timestamp}.csv", index=False)