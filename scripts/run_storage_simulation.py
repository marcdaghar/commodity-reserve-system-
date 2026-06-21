#!/usr/bin/env python3
"""
Run Yusufian storage simulation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
from src.yusuf_storage import YusufStorage

def run_storage_simulation():
    """
    Run Yusufian storage simulation.
    """
    print("=" * 60)
    print("Running Yusufian Storage Simulation...")
    print("=" * 60)
    
    # Initialize model
    model = YusufStorage(
        eta=0.7,
        C_base=50.0,
        Y_high=60.0,
        Y_low=40.0,
        S_max=1000.0,
        S_min=100.0
    )
    
    # Run simulation
    results = model.simulate(S0=500.0, T=50.0, dt=0.01)
    
    # Compute metrics
    min_stock = np.min(results['S'])
    max_stock = np.max(results['S'])
    final_stock = results['S'][-1]
    
    print("\nStorage Simulation Results:")
    print(f"  Min stock: {min_stock:.1f}")
    print(f"  Max stock: {max_stock:.1f}")
    print(f"  Final stock: {final_stock:.1f}")
    print(f"  Stock range: {max_stock - min_stock:.1f}")
    
    # Compute default probability
    print("\nComputing default probability...")
    default_prob = model.compute_default_probability(n_simulations=500, T=50.0)
    print(f"  Default probability: {default_prob:.2%}")
    
    # Save results
    os.makedirs('data', exist_ok=True)
    with open('data/storage_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results, default_prob

def main():
    """
    Main execution function.
    """
    storage_results, default_prob = run_storage_simulation()
    
    print("\n" + "=" * 60)
    print("Results saved.")
    print("=" * 60)

if __name__ == "__main__":
    main()
