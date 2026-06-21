#!/usr/bin/env python3
"""
Run CRD simulation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
from src.crd_model import CRDModel
from src.visualization import FigureGenerator

def run_crd_simulation():
    """
    Run CRD simulation.
    """
    print("=" * 60)
    print("Running CRD Simulation...")
    print("=" * 60)
    
    # Initialize model
    model = CRDModel(n_commodities=3)
    
    # Set parameters for commodities: [Gold, Wheat, Energy]
    model.floor = np.array([100.0, 80.0, 120.0])
    model.ceiling = np.array([120.0, 100.0, 150.0])
    model.alpha = np.array([0.1, 0.08, 0.12])
    model.beta = np.array([0.1, 0.08, 0.12])
    model.lambda_price = np.array([0.05, 0.04, 0.06])
    model.delta = np.array([0.01, 0.005, 0.015])
    model.mu = 0.02
    model.V_max = np.array([1000, 800, 1200])
    model.V_min = np.array([100, 80, 120])
    
    # Initial conditions
    P0 = (model.floor + model.ceiling) / 2
    V0 = model.V_min + (model.V_max - model.V_min) / 2
    M0 = 1000.0
    
    # Run simulation
    results = model.simulate(P0=P0, V0=V0, M0=M0, T=50.0, dt=0.01)
    
    # Compute metrics
    metrics = model.compute_metrics(results)
    
    print("\nCRD Simulation Results:")
    print(f"  Price variance: {metrics['price_var']}")
    print(f"  Price range: {metrics['price_range']}")
    print(f"  Stock utilization: {metrics['stock_utilization']}")
    print(f"  Money growth: {metrics['money_growth']:.2%}")
    
    # Save results
    os.makedirs('data', exist_ok=True)
    with open('data/crd_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results

def run_demurrage_comparison():
    """
    Run comparison with and without demurrage.
    """
    print("\n" + "=" * 60)
    print("Running Demurrage Comparison...")
    print("=" * 60)
    
    model = CRDModel(n_commodities=3)
    model.floor = np.array([100.0, 80.0, 120.0])
    model.ceiling = np.array([120.0, 100.0, 150.0])
    model.alpha = np.array([0.1, 0.08, 0.12])
    model.beta = np.array([0.1, 0.08, 0.12])
    model.lambda_price = np.array([0.05, 0.04, 0.06])
    model.delta = np.array([0.01, 0.005, 0.015])
    model.V_max = np.array([1000, 800, 1200])
    model.V_min = np.array([100, 80, 120])
    
    P0 = (model.floor + model.ceiling) / 2
    V0 = model.V_min + (model.V_max - model.V_min) / 2
    M0 = 1000.0
    
    # With demurrage
    model.mu = 0.02
    results_with = model.simulate(P0=P0, V0=V0, M0=M0, T=50.0, dt=0.01)
    
    # Without demurrage
    model.mu = 0.0
    results_without = model.simulate(P0=P0, V0=V0, M0=M0, T=50.0, dt=0.01)
    
    demurrage_results = {
        'M_with_demurrage': results_with['M'],
        'M_no_demurrage': results_without['M'],
        't': results_with['t']
    }
    
    # Save results
    with open('data/demurrage_results.pkl', 'wb') as f:
        pickle.dump(demurrage_results, f)
    
    return demurrage_results

def main():
    """
    Main execution function.
    """
    crd_results = run_crd_simulation()
    demurrage_results = run_demurrage_comparison()
    
    print("\n" + "=" * 60)
    print("Results saved to data/ directory.")
    print("=" * 60)

if __name__ == "__main__":
    main()
