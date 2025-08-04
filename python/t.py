import argparse

class Config:
    def __init__(self):
        self.num_runs = 1
        self.dails = False
        self.generations = 100
        self.pop_size = 100
        self.worst_size = 1
        self.elite_size = 1
        self.mutation_rate = 0.1
        self.diversity_threshold = 5000
        self.threshold_decay_rate = 0.1
        self.sigma = 30
        self.alpha = 1

config = Config()

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Test GA configuration with command line arguments')
    
    # Add arguments
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name')
    parser.add_argument('--use_dials', type=str, default='False', help='Use DIALS algorithm (True/False)')
    parser.add_argument('--generations', type=int, default=10, help='Number of generations')
    parser.add_argument('--pop_size', type=int, default=20, help='Population size')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of runs')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Convert string to boolean for use_dials
    use_dials_bool = args.use_dials.lower() == 'true'
    
    # Update global config from command line arguments
    config.dails = use_dials_bool
    config.generations = args.generations
    config.pop_size = args.pop_size
    config.num_runs = args.num_runs
    
    # Print everything to confirm arguments are working
    print("=" * 60)
    print("COMMAND LINE ARGUMENTS TEST")
    print("=" * 60)
    print(f"Experiment name: {args.experiment}")
    print(f"Use DIALS (original arg): {args.use_dials}")
    print(f"Use DIALS (converted): {use_dials_bool}")
    print(f"Generations: {args.generations}")
    print(f"Population size: {args.pop_size}")
    print(f"Number of runs: {args.num_runs}")
    
    print("\n" + "=" * 60)
    print("UPDATED CONFIG OBJECT")
    print("=" * 60)
    print(f"config.dails: {config.dails}")
    print(f"config.generations: {config.generations}")
    print(f"config.pop_size: {config.pop_size}")
    print(f"config.num_runs: {config.num_runs}")
    print(f"config.mutation_rate: {config.mutation_rate}")
    print(f"config.sigma: {config.sigma}")
    
    print("\n" + "=" * 60)
    print("SIMULATING EXPERIMENT RUNS")
    print("=" * 60)
    
    for run in range(config.num_runs):
        print(f"\n--- Simulated Run {run+1}/{config.num_runs} ---")
        print(f"  Experiment: {args.experiment}")
        print(f"  Using DIALS: {config.dails}")
        print(f"  Generations: {config.generations}")
        print(f"  Population: {config.pop_size}")
        print(f"  This would run the GA with these settings...")
    
    print(f"\n✓ Test completed for experiment: {args.experiment}")
    print("=" * 60)