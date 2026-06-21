#!/usr/bin/env python3
"""
Generate all figures for the CRD article.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
from src.visualization import FigureGenerator

def generate_all_figures():
    """
    Load saved results and generate all figures.
    """
    print("Loading saved results...")
    
    # Load CRD results
    with open('data/crd_results.pkl', 'rb') as f:
        crd_results = pickle.load(f)
    
    # Load storage results
    with open('data/storage_results.pkl', 'rb') as f:
        storage_results = pickle.load(f)
    
    # Load demurrage results
    with open('data/demurrage_results.pkl', 'rb') as f:
        demurrage_results = pickle.load(f)
    
    # Load optimization results
    with open('data/optimization_results.pkl', 'rb') as f:
        opt_results = pickle.load(f)
    
    # Generate figures
    print("Generating all figures...")
    fig_gen = FigureGenerator()
    fig_gen.figure_all(
        crd_results,
        storage_results,
        demurrage_results,
        opt_results['demand_nodes'],
        opt_results['selected_locations'],
        opt_results['potential_locations']
    )
    
    print("All figures generated successfully!")

if __name__ == "__main__":
    generate_all_figures()
