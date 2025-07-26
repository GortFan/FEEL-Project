import numpy as np
import scipy as sp
from shape2d import rectangle_mask
import random

def init_random():
    chrom = []
    for min_val, max_val in struct:
        val = random.randint(min_val, max_val)
        chrom.append(val)
    return chrom

def split_chromosome(chrom):
    return list(chrom)

def uniform_crossover(parent1, parent2):
    genes1 = split_chromosome(parent1)
    genes2 = split_chromosome(parent2)
    child = [random.choice([g1, g2]) for g1, g2 in zip(genes1, genes2)]
    return child

def mutate(chromosome):
    genes = split_chromosome(chromosome)
    idx = random.randint(0, len(genes) - 1)
    min_val, max_val = struct[idx]
    genes[idx] = random.randint(min_val, max_val)
    return genes

def elitism(generation, num_elites):
    sorted_gen = sorted(generation, key=lambda x: x[1], reverse=True)
    elites = [chrom for chrom, fit in sorted_gen[:num_elites]]
    return elites

def eval_fitness(chrom):
    # part 1
    m = np.ones((50, 200))
    params = split_chromosome(chrom)
    mask = rectangle_mask(m, 0, 50, params[0], params[1], params[2], params[3] / 10, params[4] / 10)
    m[mask] = 0
    edt = sp.ndimage.distance_transform_edt(m)
    nonzero_edt = edt[m > 0]
    avg_edt = np.mean(nonzero_edt) if nonzero_edt.size > 0 else 0
    
    # part 2
    row_to_append = 49  
    if m.shape[0] > row_to_append:
        new_row = np.ones((1, m.shape[1]))
        m = np.insert(m, row_to_append + 1, new_row, axis=0) # use axis=1 and adjust indices accordingly for the other axis
    
    # Remove all rows and columns that are all zeros
    nonzero_rows = np.any(m != 0, axis=1)
    nonzero_cols = np.any(m != 0, axis=0)
    m2 = m[nonzero_rows][:, nonzero_cols]
    edt2 = sp.ndimage.distance_transform_edt(m2)
    nonzero_edt2 = edt2[m2 > 0]
    avg_edt2 = np.mean(nonzero_edt2) if nonzero_edt2.size > 0 else 0
    
    return 0.1*avg_edt + 0.9*avg_edt2

def select_parents(population, fitnesses):
    total_fitness = sum(fitnesses)
    if total_fitness == 0:
        return random.choice(population), random.choice(population)
    probs = [f / total_fitness for f in fitnesses]
    idxs = np.random.choice(len(population), size=2, p=probs, replace=True)
    return population[idxs[0]], population[idxs[1]]

if __name__ == "__main__":
    # struct: (min_val, max_val) for each gene
    struct = [
        (6, 32),   # Example: height
        (24, 156),   # Example: width
        (0, 25),   # Example: angle
        (4, 10),   # Example: taper_x
        (4, 10),   # Example: taper_y
    ]

    POP_SIZE = 100
    NUM_GENERATIONS = 100
    ELITE_COUNT = 1
    MUTATION_RATE = 0.3

    population = [init_random() for _ in range(POP_SIZE)]
    best_so_far = float('inf')
    for gen in range(NUM_GENERATIONS):
        fitnesses = [eval_fitness(chrom) for chrom in population]
        current_best = min(fitnesses)
        if current_best < best_so_far:
            best_so_far = current_best
        print(f"Generation {gen+1}: Best-so-far fitness = {best_so_far}")
        generation = list(zip(population, fitnesses))

        elites = elitism(generation, ELITE_COUNT)

        new_population = elites.copy()
        while len(new_population) < POP_SIZE:
            parent1, parent2 = select_parents(population, fitnesses)
            child = uniform_crossover(parent1, parent2)
            if random.random() < MUTATION_RATE:
                child = mutate(child)
            new_population.append(child)

        population = new_population

        best_fitness = min(fitnesses)
        print(f"Generation {gen+1}: Best fitness = {best_fitness}")

    best_idx = np.argmin([eval_fitness(chrom) for chrom in population])
    best_chrom = population[best_idx]
    print("Best chromosome:", best_chrom)
    m = np.ones((50, 200))
    params = split_chromosome(best_chrom)
    mask = rectangle_mask(m, 0, 100, params[0], params[1], params[2], params[3] / 10, params[4] / 10)
    m[mask] = 0

    import matplotlib.pyplot as plt
    plt.imshow(m, cmap='gray')
    plt.title("Best Rectangle Mask (Ones to Zeros)")
    plt.show()