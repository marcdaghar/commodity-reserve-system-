"""
Multi-objective optimization for storage facility location.
"""

import numpy as np
import networkx as nx
from shapely.geometry import Point, Polygon
from scipy.spatial import distance_matrix

class StorageOptimizer:
    """
    Genetic algorithm for optimal storage facility location.
    """
    
    def __init__(self, demand_nodes, potential_locations,
                 capacities, w1=0.5, w2=0.3, w3=0.2,
                 pop_size=100, n_generations=100):
        """
        Args:
            demand_nodes: List of (x, y, demand) tuples
            potential_locations: List of (x, y, geo_risk, clim_risk) tuples
            capacities: Storage capacities for each potential location
            w1, w2, w3: Objective weights
            pop_size: Population size for GA
            n_generations: Number of generations
        """
        self.demand_nodes = np.array(demand_nodes)
        self.potential_locations = np.array(potential_locations)
        self.capacities = np.array(capacities)
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.pop_size = pop_size
        self.n_generations = n_generations
        
        self.n_demand = len(demand_nodes)
        self.n_locations = len(potential_locations)
        
        # Compute distance matrix
        self.dist_matrix = distance_matrix(
            demand_nodes[:, :2],
            potential_locations[:, :2]
        )
    
    def objective(self, solution):
        """
        Objective function.
        
        Args:
            solution: Binary vector indicating facility selection
        
        Returns:
            Objective value (minimize)
        """
        # Transportation cost
        transport = 0
        for i in range(self.n_demand):
            # Find nearest selected facility
            nearest_dist = np.inf
            for j in range(self.n_locations):
                if solution[j] == 1:
                    dist = self.dist_matrix[i, j]
                    if dist < nearest_dist:
                        nearest_dist = dist
            transport += self.demand_nodes[i, 2] * nearest_dist
        
        # Geopolitical risk
        geo_risk = np.sum(solution * self.potential_locations[:, 2])
        
        # Climate risk
        clim_risk = np.sum(solution * self.potential_locations[:, 3])
        
        # Total objective
        F = self.w1 * transport + self.w2 * geo_risk + self.w3 * clim_risk
        
        # Penalty for capacity violations
        for j in range(self.n_locations):
            if solution[j] == 1:
                capacity = self.capacities[j]
                assigned_demand = 0
                for i in range(self.n_demand):
                    if self._is_nearest(i, j, solution):
                        assigned_demand += self.demand_nodes[i, 2]
                if assigned_demand > capacity:
                    F += 1000 * (assigned_demand - capacity) / capacity
        
        return F
    
    def _is_nearest(self, demand_idx, location_idx, solution):
        """Check if location is nearest selected facility for given demand node."""
        dist_to_location = self.dist_matrix[demand_idx, location_idx]
        for j in range(self.n_locations):
            if solution[j] == 1 and j != location_idx:
                if self.dist_matrix[demand_idx, j] < dist_to_location:
                    return False
        return True
    
    def initialize_population(self):
        """Initialize population with random solutions."""
        population = []
        for _ in range(self.pop_size):
            # Random binary vector with at least 1 facility
            solution = np.zeros(self.n_locations)
            n_selected = np.random.randint(1, max(2, self.n_locations // 4))
            indices = np.random.choice(self.n_locations, n_selected, replace=False)
            solution[indices] = 1
            population.append(solution)
        return population
    
    def crossover(self, parent1, parent2):
        """Uniform crossover."""
        child = np.zeros(self.n_locations)
        for i in range(self.n_locations):
            if np.random.random() < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
        # Ensure at least one facility
        if np.sum(child) == 0:
            child[np.random.randint(self.n_locations)] = 1
        return child
    
    def mutate(self, solution, mutation_rate=0.05):
        """Bit-flip mutation."""
        for i in range(self.n_locations):
            if np.random.random() < mutation_rate:
                solution[i] = 1 - solution[i]
        # Ensure at least one facility
        if np.sum(solution) == 0:
            solution[np.random.randint(self.n_locations)] = 1
        return solution
    
    def select_tournament(self, population, fitness, k=3):
        """Tournament selection."""
        indices = np.random.choice(len(population), k, replace=False)
        best_idx = indices[np.argmin(fitness[indices])]
        return population[best_idx]
    
    def optimize(self):
        """
        Run genetic algorithm optimization.
        
        Returns:
            Best solution and history
        """
        # Initialize
        population = self.initialize_population()
        best_solution = None
        best_fitness = np.inf
        history = []
        
        for generation in range(self.n_generations):
            # Evaluate fitness
            fitness = np.array([self.objective(sol) for sol in population])
            
            # Track best
            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_solution = population[gen_best_idx].copy()
            
            history.append(best_fitness)
            
            # Progress
            if generation % 10 == 0:
                print(f"Generation {generation}: best = {best_fitness:.2f}")
            
            # Selection and reproduction
            new_population = []
            
            # Elitism: keep best
            new_population.append(best_solution.copy())
            
            while len(new_population) < self.pop_size:
                parent1 = self.select_tournament(population, fitness)
                parent2 = self.select_tournament(population, fitness)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            population = new_population
        
        return best_solution, best_fitness, history
    
    def compute_metrics(self, solution):
        """
        Compute resilience metrics for a solution.
        """
        n_selected = np.sum(solution)
        
        # Redundancy
        redundancy = 1 - n_selected / self.n_locations
        
        # Diversity (Shannon entropy of demand distribution)
        total_demand = np.sum(self.demand_nodes[:, 2])
        demand_fractions = []
        for j in range(self.n_locations):
            if solution[j] == 1:
                assigned_demand = 0
                for i in range(self.n_demand):
                    if self._is_nearest(i, j, solution):
                        assigned_demand += self.demand_nodes[i, 2]
                if assigned_demand > 0:
                    demand_fractions.append(assigned_demand / total_demand)
        
        if demand_fractions:
            diversity = -np.sum([p * np.log(p) for p in demand_fractions])
        else:
            diversity = 0
        
        # Concentration (max fraction)
        concentration = max(demand_fractions) if demand_fractions else 0
        
        return {
            'n_facilities': n_selected,
            'redundancy': redundancy,
            'diversity': diversity,
            'concentration': concentration
        }
