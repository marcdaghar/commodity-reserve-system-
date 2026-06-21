#!/usr/bin/env python3
"""
Run storage facility optimization.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
from src.optimization import StorageOptimizer

def generate_test_data():
    """
    Generate test data for optimization.
    """
    np.random.seed(42)
    
    # Demand nodes: 50 cities with random positions and demands
    n_demand = 50
    demand_nodes = np.zeros((n_demand, 3))
    for i in range(n_demand):
        demand_nodes[i, 0] = np.random.uniform(0, 10)  # x
        demand_nodes[i, 1] = np.random.uniform(0, 10)  # y
        demand_nodes[i, 2] = np.random.uniform(10, 50)  # demand
    
    # Potential locations: 20 sites with risks
    n_locations = 20
    potential_locations = np.zeros((n_locations, 4))
    capacities = np.zeros(n_locations)
    for i in range(n_locations):
        potential_locations[i, 0] = np.random.uniform(0, 10)  # x
        potential_locations[i, 1] = np.random.uniform(0, 10)  # y
        potential_locations[i, 2] = np.random.uniform(0, 1)  # geo risk
        potential_locations[i, 3] = np.random.uniform(0, 1)  # clim risk
        capacities[i] = np.random.uniform(200, 800)  # capacity
    
    return demand_nodes, potential_locations, capacities

def run_optimization():
    """
    Run storage facility optimization.
    """
    print("=" * 60)
    print("Running Storage Optimization...")
    print("=" * 60)
    
    # Generate test data
    demand_nodes, potential_locations, capacities = generate_test_data()
    
    # Initialize optimizer
    optimizer = StorageOptimizer(
        demand_nodes=demand_nodes,
        potential_locations=potential_locations,
        capacities=capacities,
        w1=0.5, w2=0.3, w3=0.2,
        pop_size=100,
        n_generations=50
    )
    
    # Run optimization
    best_solution, best_fitness, history = optimizer.optimize()
    
    # Get selected locations
    selected_indices = np.where(best_solution == 1)[0]
    selected_locations = potential_locations[selected_indices]
    
    # Compute metrics
    metrics = optimizer.compute_metrics(best_solution)
    
    print("\nOptimization Results:")
    print(f"  Best fitness: {best_fitness:.2f}")
    print(f"  Selected facilities: {len(selected_indices)}")
    print(f"  Redundancy: {metrics['redundancy']:.3f}")
    print(f"  Diversity: {metrics['diversity']:.3f}")
    print(f"  Concentration: {metrics['concentration']:.3f}")
    
    # Save results
    os.makedirs('data', exist_ok=True)
    results = {
        'demand_nodes': demand_nodes,
        'potential_locations': potential_locations,
        'selected_locations': selected_locations,
        'best_solution': best_solution,
        'best_fitness': best_fitness,
        'history': history,
        'metrics': metrics
    }
    with open('data/optimization_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results

def main():
    """
    Main execution function.
    """
    results = run_optimization()
    
    print("\n" + "=" * 60)
    print("Results saved to data/optimization_results.pkl")
    print("=" * 60)

if __name__ == "__main__":
    main()
