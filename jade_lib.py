import numpy as np

class JADE:
    def __init__(self, bounds, pop_size=20, c=0.1, p=0.05):
        """
        bounds: list of tuples [(min, max), ...] for each parameter
        c: learning rate for adaptive parameters
        p: top percentage for 'current-to-pbest'
        """
        self.bounds = np.array(bounds)
        self.pop_size = pop_size
        self.dim = len(bounds)
        self.c = c
        self.p = p

        # JADE Adaptive Parameters
        self.mu_cr = 0.5
        self.mu_f = 0.5
        self.archive = []

    def initialize_population(self):
        # Random initialization within bounds
        min_b, max_b = self.bounds[:, 0], self.bounds[:, 1]
        pop = min_b + np.random.rand(self.pop_size, self.dim) * (max_b - min_b)
        return pop

    def propose_next_gen(self, population, fitness_scores, generation_idx):
        """
        Generates the trial vectors for the next generation.
        Returns: list of trial vectors, list of (CR, F) pairs used
        """
        # Sort population by fitness (descending, assuming higher is better)
        sorted_idx = np.argsort(fitness_scores)[::-1]
        
        # Select the top p% individuals to form the "best" pool
        num_best = int(self.pop_size * self.p)
        # Ensure at least 1 individual is selected if p is very small
        num_best = max(1, num_best)
        
        current_best = population[sorted_idx[:num_best]]

        trials = []
        gen_params = [] # Store CR and F for each individual to update later

        for i in range(self.pop_size):
            # 1. Adapt Parameters (CR and F) for this individual
            cr = np.clip(np.random.normal(self.mu_cr, 0.1), 0, 1)
            
            f = np.clip(self.mu_f + 0.1 * np.random.standard_cauchy(), 0, 1)
            
            gen_params.append((cr, f))

            # 2. Mutation: DE/current-to-pbest/1
            # x_best_p is a random choice from the top p%
            
            if len(current_best) == 0:
                print(f"Warning: No successful individuals found in current_best (Gen {generation_idx}). Using full population.")
                current_best = population
                
            idx = np.random.randint(len(current_best))
            x_best_p = current_best[idx]
            
            x_r1 = population[np.random.randint(self.pop_size)]

            # x_r2 is chosen from (Population Union Archive)
            pop_archive = np.vstack((population, self.archive)) if len(self.archive) > 0 else population
            x_r2 = pop_archive[np.random.randint(len(pop_archive))]

            # The Mutation Equation
            x_i = population[i]
            v_i = x_i + f * (x_best_p - x_i) + f * (x_r1 - x_r2)

            # 3. Crossover (Binomial) & Bounds check
            j_rand = np.random.randint(self.dim)
            t_i = np.copy(x_i)

            for j in range(self.dim):
                if np.random.rand() < cr or j == j_rand:
                    t_i[j] = v_i[j]

            # Clip to bounds
            t_i = np.clip(t_i, self.bounds[:, 0], self.bounds[:, 1])
            trials.append(t_i)

        return np.array(trials), gen_params

    def update_archive(self, rejected_parents):
        """Adds bad parents to archive to maintain diversity"""
        self.archive.extend(rejected_parents)
        
        # Limit archive size to population size
        if len(self.archive) > self.pop_size:
            indices = np.random.choice(len(self.archive), self.pop_size, replace=False)
            self.archive = [self.archive[i] for i in indices]
