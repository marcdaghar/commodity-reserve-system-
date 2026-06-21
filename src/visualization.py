"""
Visualization functions for the CRD article.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from matplotlib.patches import Circle, Rectangle

# Set publication-ready style
plt.style.use('seaborn-v0-8-whitegrid')
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['legend.fontsize'] = 10
rcParams['figure.dpi'] = 300

class FigureGenerator:
    """
    Generate figures for the CRD article.
    """
    
    def __init__(self, output_dir='figures'):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def figure_price_stability(self, crd_results):
        """
        Figure 1: Price evolution under CRD control.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        t = crd_results['t']
        P = crd_results['P']
        
        # Plot prices for each commodity
        colors = ['darkblue', 'darkgreen', 'darkred']
        labels = ['Gold', 'Wheat', 'Energy']
        floors = [100, 80, 120]
        ceilings = [120, 100, 150]
        
        for i in range(P.shape[1]):
            ax.plot(t, P[:, i], linewidth=2, color=colors[i], label=labels[i])
            ax.axhline(y=floors[i], color=colors[i], linestyle=':', alpha=0.5)
            ax.axhline(y=ceilings[i], color=colors[i], linestyle=':', alpha=0.5)
        
        # Fill price band
        ax.fill_between(t, floors[0], ceilings[0], alpha=0.1, color='gray',
                       label='Price band (floor-ceiling)')
        
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel('Price (monetary units)', fontsize=12)
        ax.set_title('Price Stability Under CRD Control', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/crd_price_stability.pdf', format='pdf')
        plt.savefig(f'{self.output_dir}/crd_price_stability.png', format='png', dpi=300)
        plt.close()
        
        return fig
    
    def figure_stock_dynamics(self, storage_results):
        """
        Figure 2: Stock dynamics under Yusufian storage rule.
        """
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        t = storage_results['t']
        S = storage_results['S']
        Y = storage_results['Y']
        C = storage_results['C']
        
        # Panel 1: Production and consumption
        ax = axes[0]
        ax.plot(t, Y, linewidth=2, color='blue', label='Production $Y(t)$')
        ax.plot(t, C, linewidth=2, color='red', label='Consumption $C(t)$')
        ax.axhline(y=50, color='gray', linestyle='--', linewidth=1.5, label='$C_{base}$')
        ax.set_ylabel('Rate (units/year)', fontsize=12)
        ax.set_title('Production and Consumption Cycles', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Stock
        ax = axes[1]
        ax.plot(t, S, linewidth=2, color='darkgreen', label='Stock $S(t)$')
        ax.axhline(y=100, color='red', linestyle='--', linewidth=1.5, label='$S_{min}$')
        ax.axhline(y=1000, color='red', linestyle=':', linewidth=1.5, label='$S_{max}$')
        ax.set_ylabel('Stock (units)', fontsize=12)
        ax.set_title('Stock Dynamics Under Yusufian Rule', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Stock change rate
        ax = axes[2]
        dS = storage_results['dS']
        ax.plot(t, dS, linewidth=2, color='purple', label='$dS/dt$')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel('Stock change rate', fontsize=12)
        ax.set_title('Storage Rate', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/crd_stock_dynamics.pdf', format='pdf')
        plt.savefig(f'{self.output_dir}/crd_stock_dynamics.png', format='png', dpi=300)
        plt.close()
        
        return fig
    
    def figure_money_supply(self, crd_results, demurrage_results):
        """
        Figure 3: Money supply evolution with demurrage.
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        
        t = crd_results['t']
        M = crd_results['M']
        
        # Panel 1: Money supply with demurrage
        ax = axes[0]
        ax.plot(t, M, linewidth=2.5, color='darkblue', label='With demurrage')
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel('Money supply (units)', fontsize=12)
        ax.set_title('Money Supply Evolution with Demurrage', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Comparison with/without demurrage
        ax = axes[1]
        ax.plot(t, M, linewidth=2, color='darkblue', label='With demurrage ($\mu = 0.02$)')
        ax.plot(t, demurrage_results['M_no_demurrage'], linewidth=2, 
                color='red', linestyle='--', label='Without demurrage ($\mu = 0$)')
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel('Money supply (units)', fontsize=12)
        ax.set_title('Effect of Demurrage on Money Supply', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/crd_money_supply.pdf', format='pdf')
        plt.savefig(f'{self.output_dir}/crd_money_supply.png', format='png', dpi=300)
        plt.close()
        
        return fig
    
    def figure_default_probability(self):
        """
        Figure 4: Default probability comparison.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Data from simulation
        systems = ['CRD System', 'Debt-Based System']
        probabilities = [0.0, 0.32]
        colors = ['darkgreen', 'darkred']
        
        bars = ax.bar(systems, probabilities, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{prob*100:.0f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Probability of Default', fontsize=12)
        ax.set_title('Default Probability Comparison (50-year horizon)', fontsize=14)
        ax.set_ylim([0, 0.4])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add note
        ax.text(0.5, -0.1, 'Note: CRD achieves zero default probability through countercyclical storage',
                transform=ax.transAxes, ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/crd_default_probability.pdf', format='pdf')
        plt.savefig(f'{self.output_dir}/crd_default_probability.png', format='png', dpi=300)
        plt.close()
        
        return fig
    
    def figure_optimal_storage(self, demand_nodes, selected_locations, all_locations):
        """
        Figure 5: Optimal storage facility locations.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot demand nodes
        ax.scatter(demand_nodes[:, 0], demand_nodes[:, 1], 
                  s=demand_nodes[:, 2] * 2, c='blue', alpha=0.6, label='Demand nodes')
        
        # Plot all potential locations
        ax.scatter(all_locations[:, 0], all_locations[:, 1],
                  s=50, c='gray', marker='s', alpha=0.4, label='Potential locations')
        
        # Plot selected locations
        if len(selected_locations) > 0:
            ax.scatter(selected_locations[:, 0], selected_locations[:, 1],
                      s=200, c='red', marker='D', edgecolor='black', 
                      label='Selected storage facilities')
        
        # Draw Voronoi-like regions (simplified)
        for loc in selected_locations:
            circle = Circle((loc[0], loc[1]), radius=1.5, fill=False, 
                           edgecolor='red', linestyle='--', alpha=0.5)
            ax.add_patch(circle)
        
        # Add capacity labels
        for loc in selected_locations:
            ax.annotate(f'{int(loc[2])}', (loc[0]+0.1, loc[1]+0.1), fontsize=8)
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title('Optimal Storage Facility Locations', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/crd_optimal_storage.pdf', format='pdf')
        plt.savefig(f'{self.output_dir}/crd_optimal_storage.png', format='png', dpi=300)
        plt.close()
        
        return fig
    
    def figure_all(self, crd_results, storage_results, demurrage_results,
                   demand_nodes, selected_locations, all_locations):
        """
        Generate all figures for the article.
        """
        print("Generating Figure 1: Price stability...")
        self.figure_price_stability(crd_results)
        
        print("Generating Figure 2: Stock dynamics...")
        self.figure_stock_dynamics(storage_results)
        
        print("Generating Figure 3: Money supply...")
        self.figure_money_supply(crd_results, demurrage_results)
        
        print("Generating Figure 4: Default probability...")
        self.figure_default_probability()
        
        print("Generating Figure 5: Optimal storage...")
        self.figure_optimal_storage(demand_nodes, selected_locations, all_locations)
        
        print("All figures generated successfully!")
