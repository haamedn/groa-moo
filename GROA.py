# Greedy Respiration Optimization Algorithm (GROA)
# Multi-Objective Optimization Version with Complex Constraints and Real Engineering Test, By Hamed Nozari

import numpy as np
import matplotlib.pyplot as plt

# Objective functions (Engineering-inspired multi-objective)
def objective_1(x):
    # Minimize weight
    return x[0]**2 + x[1]**2 + x[2]**2

def objective_2(x):
    # Minimize cost or power loss
    return (x[0]-2)**2 + (x[1]-1)**2 + x[2]**2

# Complex constraint handling using penalty method
def penalty(x):
    penalties = []
    # Constraint 1: x[0] + x[1] <= 2.5
    penalties.append(max(0, x[0] + x[1] - 2.5))
    # Constraint 2: x[2] >= 0.5
    penalties.append(max(0, 0.5 - x[2]))
    return sum(penalties)

# Initialization
def initialize_population(pop_size, dim):
    return np.random.rand(pop_size, dim)

# Evaluation with penalties
def evaluate_population(pop):
    obj_vals = []
    feasibles = []
    for ind in pop:
        f1 = objective_1(ind)
        f2 = objective_2(ind)
        pen = penalty(ind)
        if pen == 0:
            feasibles.append(True)
        else:
            # Add penalty to objective values to degrade infeasible solutions
            f1 += 1000 * pen
            f2 += 1000 * pen
            feasibles.append(False)
        obj_vals.append([f1, f2])
    return np.array(obj_vals), np.array(feasibles)

# Non-dominated sorting
def pareto_sort(objectives, feasibles):
    fronts = []
    remaining = list(range(len(objectives)))
    while remaining:
        front = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i != j and dominates(objectives[j], objectives[i]):
                    dominated = True
                    break
            if not dominated:
                front.append(i)
        fronts.append(front)
        remaining = [i for i in remaining if i not in front]
    return fronts

def dominates(a, b):
    return all(a <= b) and any(a < b)

# Inflow Phase (exploration)
def inflow_phase(pop, pressure):
    return pop + pressure * np.random.uniform(-1, 1, pop.shape)

# Outflow Phase (exploitation)
def outflow_phase(pop, best, pressure):
    return pop + pressure * (best - pop) * np.random.rand(*pop.shape)

# Adaptive pressure update
def update_pressure(iteration, max_iter):
    return 1 - (iteration / max_iter)

# Main GROA loop
def groa(pop_size=100, dim=3, max_iter=200):
    pop = initialize_population(pop_size, dim)
    best_archive = []

    for t in range(max_iter):
        pressure = update_pressure(t, max_iter)
        obj_vals, feasibles = evaluate_population(pop)
        fronts = pareto_sort(obj_vals, feasibles)
        best_indices = fronts[0]
        best_solutions = pop[best_indices]
        best = best_solutions[np.random.randint(len(best_solutions))]

        # Inflow
        pop = inflow_phase(pop, pressure)
        pop = np.clip(pop, 0, 1)

        # Pressure update
        pressure = update_pressure(t, max_iter)

        # Outflow
        pop = outflow_phase(pop, best, pressure)
        pop = np.clip(pop, 0, 1)

        # Update archive
        best_archive = pop[best_indices]

    final_obj, _ = evaluate_population(best_archive)
    return best_archive, final_obj

# Run the algorithm and plot Pareto front
solutions, objectives = groa()
plt.scatter(objectives[:, 0], objectives[:, 1], c='red', label='GROA Pareto Front')
plt.xlabel('Objective 1: Min Weight')
plt.ylabel('Objective 2: Min Cost')
plt.title('GROA Pareto Front on Engineering Design Problem')
plt.legend()
plt.grid(True)
plt.show()
