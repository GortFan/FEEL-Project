import sys, os, time, math, random
import numpy as np
from shape3d import elliptical_cylinder_mask, cuboid_mask
import pyvista as pv
from multiprocessing import Process, Queue
import cupy as cp
import cupyx.scipy.ndimage as cp_ndimage
import pandas as pd
import argparse

project_root = os.path.dirname(os.path.dirname(__file__))

sys.path.append(os.path.join(project_root, 'build', 'Release'))

import FEELcpp as fc

SHAPE_CONFIGS = {
    'cuboid': {
        'struct': [
            (1, 60),    
            (1, 248),     
            (1, 248),    
            (0, 180),   
            (0, 180),   
            (0, 10),    
            (0, 10),    
        ],
        'param_names': ['cuboid_depth', 'cuboid_height', 'cuboid_width', 'yaw', 'pitch', 'taper_width', 'taper_height']
    },
    'ellipsoid': {
        'struct': [
            (1, 60),    
            (1, 124),    
            (1, 124),      
            (0, 180),   
            (0, 180),   
            (0, 10),   
            (0, 0),     # unused (keep 7 params for consistency)
        ],
        'param_names': ['cylinder_length', 'radius_y', 'radius_x', 'yaw', 'pitch', 'taper_z', 'unused']
    }
}

class Config:
    def __init__(self):
        self.generations = 100
        self.pop_size = 100
        self.worst_size = 1
        self.elite_size = 1
        self.mutation_rate = 0.1
        self.diversity_threshold = 5000
        self.threshold_decay_rate = 0.1
        self.sigma = 30
        self.alpha = 1
        self.num_processes = 5 
        self.shape = 'cuboid'  
        
    @property
    def struct(self):
        return SHAPE_CONFIGS[self.shape]['struct']
    
    @property  
    def param_names(self):
        return SHAPE_CONFIGS[self.shape]['param_names']

config = Config()

def eval1(m) -> float:
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
    if usable_c < math.ceil(m.shape[0]*m.shape[1]*m.shape[2]*0.50):
        penalty = (m.shape[0]*m.shape[1]*m.shape[2] - usable_c) + (m.shape[0]*m.shape[1]*m.shape[2] // 100)
    f = e1 + e2 + penalty
    return f

def initialize_pop(pop_size: int) -> list:
    bounds = np.array(config.struct)  # Use config.struct instead of global struct
    lows = bounds[:, 0]
    highs = bounds[:, 1] + 1
    
    pop = np.random.randint(low=lows[:, None], high=highs[:, None], size=(len(config.struct), pop_size)).T
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
    min_val, max_val = config.struct[idx]  # Use config.struct
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
    
def create_ridge(m_shape, individual_batch, process_index, shape_type, queue):
    result = [None]*len(individual_batch)
    for idx, individual in enumerate(individual_batch):
        mc = np.ones(m_shape)
        mc[0, :, :] = 0
        if shape_type == 'cuboid':
            print("cubers")
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
                            taper_width=individual[5] / 10,
                            taper_height=individual[6] / 10)
        
        elif shape_type == 'ellipsoid':
            print("ellipt")
            r = elliptical_cylinder_mask(matrix=mc,
                                        base_z=0,
                                        base_y=mc.shape[1] // 2,
                                        base_x=mc.shape[2] // 2,
                                        cylinder_length=individual[0],
                                        radius_y=individual[1],
                                        radius_x=individual[2],
                                        yaw=individual[3],
                                        pitch=individual[4],
                                        roll=0,
                                        taper_z=individual[5] / 10)
        else:
            raise ValueError(f"Unknown shape type: {shape_type}")
            
        mc[r] = 0
        c_req = (mc.shape[0] - 1)*mc.shape[1]*mc.shape[2]
        c_curr = np.count_nonzero(mc == 1)
        c_layer = mc.shape[1]*mc.shape[2]
        num_layer = math.ceil((c_req - c_curr) / c_layer)
        m_appended_c = np.append(mc, np.ones((num_layer, mc.shape[1], mc.shape[2])), axis=0)
        f = fitness(m_appended_c)
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
        p = Process(target=create_ridge, args=(m_shape, pop[start:end], process_index, config.shape, queue))
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
        raw_fitnesses = parallelize_ridge_evaluation(config.num_processes, np.zeros(pop_size), pop, pop_size, m.shape)
        current_min_idx = np.argmin(raw_fitnesses)
        current_min_fitness = raw_fitnesses[current_min_idx]

        if current_min_fitness < best_raw_fitness:
            best_raw_fitness = current_min_fitness
            best_chrom = pop[current_min_idx].copy()

        diversity = compute_diversity(pop)

        generation_data.append({
            'generation': generation_number,
            'best_fitness': best_raw_fitness,
            'current_min_fitness': current_min_fitness,
            'best_chromosome': best_chrom.tolist() if best_chrom is not None else None,
            'diversity': diversity
        })
        current_diversity_threshold = diversity_threshold * np.exp(-diversity_decay_rate * generation_number)
        if diversity < current_diversity_threshold and generation_number != generation_qty - 1:
            print("diversity")
            selection_fitnesses = penalize_drift_sharing(pop, raw_fitnesses, sigma, alpha)
        else:
            print("raw fitness is carried over")
            selection_fitnesses = raw_fitnesses.copy() 

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
    
    final_raw_fitnesses = parallelize_ridge_evaluation(config.num_processes, np.zeros(pop_size), pop, pop_size, m.shape)
    current_min_idx = np.argmin(raw_fitnesses)
    current_min_fitness = raw_fitnesses[current_min_idx]

    if current_min_fitness < best_raw_fitness:
        best_raw_fitness = current_min_fitness
        best_chrom = pop[current_min_idx].copy()

    diversity = compute_diversity(pop)

    generation_data.append({
        'generation': generation_number+1,
        'best_fitness': best_raw_fitness,
        'current_min_fitness': current_min_fitness,
        'best_chromosome': best_chrom.tolist() if best_chrom is not None else None,
        'diversity': diversity
    })
    
    df = pd.DataFrame(generation_data)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"Individual_Run_Fitness_History_{timestamp}.csv"
    df.to_csv(filename, index=False)

    return best_raw_fitness, best_chrom.tolist(), list(zip(pop.tolist(), final_raw_fitnesses.tolist())), fitness_history

def GA_dispatch():
    print(f"DEBUG: GA_dispatch using config - "
          f"generations={config.generations}, "
          f"pop_size={config.pop_size}, "
          f"elite_size={config.elite_size}, "
          f"worst_size={config.worst_size}, "
          f"mutation_rate={config.mutation_rate}, "
          f"diversity_threshold={config.diversity_threshold}, "
          f"threshold_decay_rate={config.threshold_decay_rate}, "
          f"sigma={config.sigma}, "
          f"alpha={config.alpha}, "
          f"num_processes={config.num_processes}")
    
    m = np.ones((61, 124, 124))
    m[0, :, :] = 0
    
    fmin, fmin_chrom, tgen, fitnesses = genetic_algorithm(
        m,
        config.generations,
        config.pop_size,
        config.elite_size,
        config.worst_size,
        config.mutation_rate,
        config.diversity_threshold,
        config.threshold_decay_rate,
        config.sigma,
        config.alpha
    )
    return fmin, fmin_chrom, tgen, fitnesses
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GA with specific configuration')
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name')
    parser.add_argument('--shape', type=str, default='cuboid', choices=['cuboid', 'ellipsoid'], help='Shape type')
    parser.add_argument('--generations', type=int, default=10, help='Number of generations')
    parser.add_argument('--pop_size', type=int, default=20, help='Population size')
    parser.add_argument('--elite_size', type=int, default=1, help='Elite size')
    parser.add_argument('--worst_size', type=int, default=1, help='Worst size')
    parser.add_argument('--mutation_rate', type=float, default=0.1, help='Mutation rate')
    parser.add_argument('--diversity_threshold', type=float, default=5000, help='Diversity threshold')
    parser.add_argument('--threshold_decay_rate', type=float, default=0.1, help='Threshold decay rate')
    parser.add_argument('--sigma', type=float, default=30, help='Sigma for sharing function')
    parser.add_argument('--alpha', type=float, default=1, help='Alpha for sharing function')
    parser.add_argument('--num_processes', type=int, default=5, help='Number of parallel processes')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of runs')
    
    args = parser.parse_args()

    config.shape = args.shape
    config.generations = args.generations
    config.pop_size = args.pop_size
    config.elite_size = args.elite_size          
    config.worst_size = args.worst_size          
    config.mutation_rate = args.mutation_rate    
    config.diversity_threshold = args.diversity_threshold  
    config.threshold_decay_rate = args.threshold_decay_rate  
    config.sigma = args.sigma                    
    config.alpha = args.alpha                    
    config.num_processes = args.num_processes    
    
    print("=" * 60)
    print(f"EXPERIMENT: {args.experiment}")
    print("=" * 60)
    print(f"config.shape: {config.shape}")
    print(f"config.generations: {config.generations}")
    print(f"config.pop_size: {config.pop_size}")
    print(f"Structure: {config.param_names}")
    print("=" * 60)
    
    all_results = []
    experiment_start_time = time.time()
    
    for run in range(args.num_runs):
        run_start_time = time.time()
        print(f"\n--- Run {run+1}/{args.num_runs} ---")
        
        seed = random.randint(1, 10000)
        np.random.seed(seed)
        random.seed(seed)
        cp.random.seed(seed)
        
        try:
            result = GA_dispatch()
            
            run_time = time.time() - run_start_time
            all_results.append({
                'experiment': args.experiment,
                'run': run + 1,
                'seed': seed,
                'generations': config.generations,
                'pop_size': config.pop_size,
                'best_fitness': result[0],
                'best_chromosome': result[1],
                'run_time_minutes': run_time / 60
            })
            
            print(f"✓ Best fitness: {result[0]:.6f}")
            print(f"✓ Runtime: {run_time/60:.1f} minutes")
            
        except Exception as e:
            print(f"✗ Run {run+1} failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Save results
    experiment_time = time.time() - experiment_start_time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    df = pd.DataFrame(all_results)
    filename = f"ga_experiment_{args.experiment}_{timestamp}.csv"
    df.to_csv(filename, index=False)
    
    print(f"\n EXPERIMENT {args.experiment} ")
    print(f"Total time: {experiment_time/60:.1f} minutes")
    print(f"Results saved to: {filename}")