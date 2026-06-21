"""
Commodity Reserve Department (CRD) Model.
Implements the core CRD dynamics with floor-ceiling price controls.
"""

import numpy as np
from scipy.integrate import odeint

class CRDModel:
    """
    Commodity Reserve Department model with price floor/ceiling controls.
    
    State variables:
        P_i(t): Price of commodity i
        V_i(t): Stock of commodity i
        M(t): Money supply
    
    Control law:
        If P_i < floor_i: Buy commodity, create money
        If P_i > ceiling_i: Sell commodity, destroy money
        Otherwise: No intervention
    """
    
    def __init__(self, n_commodities=3):
        """
        Args:
            n_commodities: Number of commodities in the basket
        """
        self.n = n_commodities
        
        # Default parameters
        self.floor = np.array([100.0, 80.0, 120.0])  # Price floors
        self.ceiling = np.array([120.0, 100.0, 150.0])  # Price ceilings
        self.alpha = np.array([0.1, 0.08, 0.12])  # Buying elasticities
        self.beta = np.array([0.1, 0.08, 0.12])  # Selling elasticities
        self.lambda_price = np.array([0.05, 0.04, 0.06])  # Price adjustment speeds
        self.delta = np.array([0.01, 0.005, 0.015])  # Storage decay rates
        self.mu = 0.02  # Demurrage rate
        self.V_max = np.array([1000, 800, 1200])  # Max storage capacities
        self.V_min = np.array([100, 80, 120])  # Min reserve requirements
    
    def control_law(self, P, V, M):
        """
        CRD control law.
        
        Args:
            P: Prices (array)
            V: Stocks (array)
            M: Money supply
        
        Returns:
            Q_buy: Purchase quantities
            Q_sell: Sell quantities
            dM: Money supply change
            dV: Stock changes
        """
        Q_buy = np.zeros(self.n)
        Q_sell = np.zeros(self.n)
        dV = np.zeros(self.n)
        dM = 0.0
        
        for i in range(self.n):
            if P[i] < self.floor[i]:
                # Abundance: buy and store
                Q_buy[i] = self.alpha[i] * (self.floor[i] - P[i])
                Q_buy[i] = min(Q_buy[i], self.V_max[i] - V[i])  # Capacity constraint
                dV[i] = Q_buy[i] - self.delta[i] * V[i]
                dM += Q_buy[i] * self.floor[i]
                
            elif P[i] > self.ceiling[i]:
                # Scarcity: sell from stock
                Q_sell[i] = min(V[i], self.beta[i] * (P[i] - self.ceiling[i]))
                Q_sell[i] = min(Q_sell[i], V[i] - self.V_min[i])  # Reserve constraint
                dV[i] = -Q_sell[i] - self.delta[i] * V[i]
                dM -= Q_sell[i] * self.ceiling[i]
                
            else:
                # Stable: no intervention
                dV[i] = -self.delta[i] * V[i]
        
        # Demurrage on money supply
        dM -= self.mu * M
        
        return Q_buy, Q_sell, dM, dV
    
    def price_dynamics(self, P, V, Q_buy, Q_sell, t):
        """
        Price dynamics with supply-demand and CRD intervention.
        
        dP_i/dt = lambda_i * (D_i - S_i - Q_buy_i + Q_sell_i)
        """
        # Simple supply and demand (can be extended)
        D = 50.0 + 5.0 * np.sin(2 * np.pi * t / 7)  # Demand with seasonal cycle
        S = 55.0 + 5.0 * np.sin(2 * np.pi * t / 7 + 1)  # Supply with lag
        
        dP = self.lambda_price * (D - S - Q_buy + Q_sell)
        return dP
    
    def system_dynamics(self, state, t):
        """
        Full system dynamics.
        
        State vector: [P_1, ..., P_n, V_1, ..., V_n, M]
        """
        n = self.n
        P = state[:n]
        V = state[n:2*n]
        M = state[2*n]
        
        # CRD control
        Q_buy, Q_sell, dM, dV = self.control_law(P, V, M)
        
        # Price dynamics
        dP = self.price_dynamics(P, V, Q_buy, Q_sell, t)
        
        # Assemble derivative
        dstate = np.concatenate([dP, dV, [dM]])
        
        return dstate
    
    def simulate(self, P0=None, V0=None, M0=1000, T=50.0, dt=0.01):
        """
        Run simulation.
        
        Args:
            P0: Initial prices
            V0: Initial stocks
            M0: Initial money supply
            T: Simulation time (years)
            dt: Time step
        
        Returns:
            Dictionary with results
        """
        if P0 is None:
            P0 = (self.floor + self.ceiling) / 2  # Initial prices at mid-point
        if V0 is None:
            V0 = self.V_min + (self.V_max - self.V_min) / 2
        
        # Time vector
        t = np.arange(0, T + dt, dt)
        n_steps = len(t)
        
        # Initialize state
        state0 = np.concatenate([P0, V0, [M0]])
        state = np.zeros((n_steps, len(state0)))
        state[0] = state0
        
        # Integration
        for i in range(n_steps - 1):
            # Current state
            P = state[i, :self.n]
            V = state[i, self.n:2*self.n]
            M = state[i, 2*self.n]
            
            # CRD control
            Q_buy, Q_sell, dM, dV = self.control_law(P, V, M)
            
            # Price dynamics
            dP = self.price_dynamics(P, V, Q_buy, Q_sell, t[i])
            
            # Euler step
            state[i+1, :self.n] = P + dP * dt
            state[i+1, self.n:2*self.n] = V + dV * dt
            state[i+1, 2*self.n] = M + dM * dt
            
            # Ensure non-negative
            state[i+1, :self.n] = np.maximum(state[i+1, :self.n], 0)
            state[i+1, self.n:2*self.n] = np.maximum(state[i+1, self.n:2*self.n], self.V_min)
            state[i+1, self.n:2*self.n] = np.minimum(state[i+1, self.n:2*self.n], self.V_max)
            state[i+1, 2*self.n] = np.maximum(state[i+1, 2*self.n], 0)
        
        # Extract results
        P = state[:, :self.n]
        V = state[:, self.n:2*self.n]
        M = state[:, 2*self.n]
        
        return {
            't': t,
            'P': P,
            'V': V,
            'M': M,
            'Q_buy': None,  # Would require storing
            'Q_sell': None
        }
    
    def compute_metrics(self, results):
        """
        Compute performance metrics.
        """
        P = results['P']
        t = results['t']
        
        # Price stability (variance within band)
        price_var = np.var(P, axis=0)
        price_range = np.max(P, axis=0) - np.min(P, axis=0)
        
        # Stock utilization
        V = results['V']
        stock_util = np.mean(V / self.V_max, axis=0)
        
        # Money supply growth
        M = results['M']
        money_growth = (M[-1] - M[0]) / M[0]
        
        return {
            'price_var': price_var,
            'price_range': price_range,
            'stock_utilization': stock_util,
            'money_growth': money_growth
        }
