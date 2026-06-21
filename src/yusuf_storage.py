"""
Yusufian Storage Rule.
Implements the countercyclical storage rule based on the seven-year cycle.
"""

import numpy as np

class YusufStorage:
    """
    Yusufian storage rule: 7 years of abundance + 7 years of scarcity.
    
    Storage rule:
        If Y > Y_high: Store fraction eta of surplus
        If Y < Y_low: Drawdown to maintain consumption
        Otherwise: No change
    """
    
    def __init__(self, eta=0.7, C_base=50.0, Y_high=60.0, Y_low=40.0,
                 S_max=1000.0, S_min=100.0):
        """
        Args:
            eta: Storage fraction in abundance (0.5 - 0.8)
            C_base: Basic consumption need
            Y_high: Production threshold for abundance
            Y_low: Production threshold for scarcity
            S_max: Maximum storage capacity
            S_min: Minimum reserve
        """
        self.eta = eta
        self.C_base = C_base
        self.Y_high = Y_high
        self.Y_low = Y_low
        self.S_max = S_max
        self.S_min = S_min
    
    def production_cycle(self, t):
        """
        Yusufian production cycle: 7 years abundance, 7 years scarcity.
        
        Args:
            t: Time in years
        
        Returns:
            Production rate
        """
        # Cycle period: 14 years
        phase = t % 14
        
        # 7 years abundance, 7 years scarcity
        if phase < 7:
            # Abundance: high production with some variation
            Y = 70.0 + 10.0 * np.sin(2 * np.pi * phase / 7)
        else:
            # Scarcity: low production
            Y = 30.0 + 10.0 * np.sin(2 * np.pi * (phase - 7) / 7)
        
        return Y
    
    def storage_rule(self, Y, S):
        """
        Storage rule based on production level.
        
        Args:
            Y: Current production
            S: Current stock
        
        Returns:
            dS: Change in stock
        """
        if Y > self.Y_high and S < self.S_max:
            # Abundance: store fraction of surplus
            surplus = Y - self.C_base
            dS = self.eta * surplus
            dS = min(dS, self.S_max - S)  # Capacity constraint
            
        elif Y < self.Y_low and S > self.S_min:
            # Scarcity: drawdown to maintain consumption
            deficit = self.C_base - Y
            dS = -deficit
            dS = max(dS, self.S_min - S)  # Reserve constraint
            
        else:
            # Normal: no change
            dS = 0.0
        
        return dS
    
    def simulate(self, S0=500.0, T=50.0, dt=0.01):
        """
        Simulate storage dynamics.
        
        Args:
            S0: Initial stock
            T: Simulation time (years)
            dt: Time step
        
        Returns:
            Dictionary with results
        """
        t = np.arange(0, T + dt, dt)
        n_steps = len(t)
        
        S = np.zeros(n_steps)
        Y = np.zeros(n_steps)
        C = np.zeros(n_steps)
        dS_array = np.zeros(n_steps)
        
        S[0] = S0
        
        for i in range(n_steps - 1):
            # Production
            Y[i] = self.production_cycle(t[i])
            
            # Storage rule
            dS_array[i] = self.storage_rule(Y[i], S[i])
            
            # Consumption
            if Y[i] > self.Y_high:
                C[i] = self.C_base
            elif Y[i] < self.Y_low:
                C[i] = min(self.C_base, S[i] + Y[i])
            else:
                C[i] = Y[i]
            
            # Update stock
            S[i+1] = S[i] + dS_array[i] * dt
            S[i+1] = np.clip(S[i+1], self.S_min, self.S_max)
        
        # Last values
        Y[-1] = self.production_cycle(t[-1])
        C[-1] = C[-2]
        
        return {
            't': t,
            'S': S,
            'Y': Y,
            'C': C,
            'dS': dS_array
        }
    
    def compute_default_probability(self, n_simulations=1000, T=50.0):
        """
        Compute probability of default over multiple simulations.
        
        Args:
            n_simulations: Number of Monte Carlo runs
            T: Simulation horizon
        
        Returns:
            Default probability
        """
        defaults = 0
        
        for _ in range(n_simulations):
            # Random initial stock
            S0 = np.random.uniform(200, 800)
            
            # Random parameters
            eta = np.random.uniform(0.5, 0.8)
            Y_high = np.random.uniform(55, 65)
            Y_low = np.random.uniform(35, 45)
            
            # Create model with random parameters
            model = YusufStorage(eta=eta, C_base=self.C_base,
                                Y_high=Y_high, Y_low=Y_low,
                                S_max=self.S_max, S_min=self.S_min)
            
            # Run simulation
            results = model.simulate(S0=S0, T=T)
            
            # Check if default occurred (stock below minimum)
            if np.min(results['S']) <= self.S_min * 0.9:
                defaults += 1
        
        return defaults / n_simulations
