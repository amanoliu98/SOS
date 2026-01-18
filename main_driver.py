import os
import pickle
import numpy as np
import pandas as pd
import subprocess
from src.jade_lib import JADE

# --- CONFIGURATION ---
CHECKPOINT_PATH = "checkpoints/jade_state.pkl"
METRICS_PATH = "metrics/history.csv"
POP_SIZE = 10
MAX_GEN = 100
GPUS = [0, 1]

# --- JADE VECTOR ---
# Index 0: Learning Rate
# Index 1: Dropout
# Index 2: LAYERS
# Index 3: unused
# Index 4: Sparsity
# Index 5: Bias
# Index 6: Quant bits
BOUNDS = [
    (1e-6, 5e-5), # 0
    (0.0, 0.5),   # 1
    (6, 24),      # 2
    (1, 3.99),    # 3
    (0.1, 0.95),  # 4
    (-1.0, 1.0),  # 5
    (4, 8)        # 6
]

def save_checkpoint(jade_obj, pop, fitness, gen):
    os.makedirs("checkpoints", exist_ok=True)
    with open(CHECKPOINT_PATH, 'wb') as f:
        pickle.dump({
            'jade': jade_obj, 
            'pop': pop, 
            'fitness': fitness, 
            'gen': gen
        }, f)
    print(f"--> [checkpoint] Saved generation {gen}", flush=True)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, 'rb') as f:
            data = pickle.load(f)
            print(f"--> [checkpoint] Resuming from generation {data['gen']}", flush=True)
            return data['jade'], data['pop'], data['fitness'], data['gen']
    return None, None, None, 0

def evaluate_population(population, generation):
    scores = np.zeros(len(population))
    
    for i in range(0, len(population), len(GPUS)):
        batch_procs = []
        batch_indices = []
        
        for gpu_idx, gpu_id in enumerate(GPUS):
            if i + gpu_idx < len(population):
                idx = i + gpu_idx
                vec_str = ",".join(map(str, population[idx]))
                
                cmd = [
                    "python", "-u", "src/worker.py", 
                    "--gpu", str(gpu_id), 
                    "--vector", vec_str, 
                    "--gen", str(generation), 
                    "--id", str(idx)
                ]
                print(f"Gen {generation} | Ind {idx} -> GPU {gpu_id}", flush=True)
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                batch_procs.append(p)
                batch_indices.append(idx)

        for p, idx in zip(batch_procs, batch_indices):
            last_line = "0.0"
            for line in p.stdout:
                print(line.strip(), flush=True)
                if "RESULT|" in line: 
                    last_line = line.strip().split("|")[-1]
            p.wait()
            try: scores[idx] = float(last_line)
            except: scores[idx] = 0.0
    return scores

def main():
    os.makedirs("metrics", exist_ok=True)
    jade, pop, fitness, start_gen = load_checkpoint()
    
    if jade is None:
        print("--> Starting Fresh Experiment", flush=True)
        jade = JADE(BOUNDS, pop_size=POP_SIZE)
        pop = jade.initialize_population()
        fitness = evaluate_population(pop, 0)
        save_checkpoint(jade, pop, fitness, 0)
    else:
        jade.bounds = np.array(BOUNDS)

    # Standard Evolutionary Loop
    for gen in range(start_gen + 1, MAX_GEN):
        print(f"\n=== GENERATION {gen} ===", flush=True)
        
        trials, params = jade.propose_next_gen(pop, fitness, gen)
        trials = np.array(trials) # Ensure numpy for clipping
        
        for i in range(len(BOUNDS)):
            trials[:, i] = np.clip(trials[:, i], BOUNDS[i][0], BOUNDS[i][1])
        
        trials = trials.tolist()
        
        trial_fitness = evaluate_population(trials, gen)
        
        for i in range(POP_SIZE):
            if trial_fitness[i] > fitness[i]:
                pop[i] = trials[i]
                fitness[i] = trial_fitness[i]
        
        save_checkpoint(jade, pop, fitness, gen)
        
        best_fit = max(fitness)
        print(f"Gen {gen} Best Fitness: {best_fit:.4f}", flush=True)
        pd.DataFrame([{'gen': gen, 'best_f': best_fit, 'avg_f': np.mean(fitness)}]).to_csv(METRICS_PATH, mode='a', header=not os.path.exists(METRICS_PATH), index=False)

if __name__ == "__main__": main()
