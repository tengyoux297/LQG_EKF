import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import scipy.linalg
import scipy.signal
import matplotlib.pyplot as plt
from typing import Literal
import tqdm
from Newton import *
from stateDynamics import *

small_value = 1e-6  # Small value to prevent numerical issues


class LQG_QKF:
    def __init__(self, F: StateDynamics, S: sensor,
                 Q, R,
                dt = 0.1, H = 50):
        
        # state settings
        self.F = F
        self.A = F.get_A()
        self.B = F.get_B()
        self.n = F.get_state_size()
        self.p = F.get_input_size()
        
        # forward dynamics
        self.F = F
        self.dt = dt
        
        # horizon
        self.H = H
        
        # states
        self.x = F.get_current_state() # current state vector
        self.x_hat = np.zeros((self.n, 1)) # estimated state vector
        self.u = F.get_current_control() # current control input vector
        
        # lqe 
        self.kalman_gain = None # kalman gain
        self.P_lqe = np.eye(self.m) * small_value  # estimation error covariance matrix
        
        # lqr
        self.Q = Q.astype(np.float64)
        self.R = R.astype(np.float64)
        self.P_lqr = self.Q
    
    def update_lqr(self, goal_state):
        x = self.x - goal_state
        z, z1, z2 = self.F.aug_state()
        
        
        
        from scipy.linalg import solve_discrete_are
        P = solve_discrete_are(A_tilde, B_tilde, self.Q, self.R)   # P is the fixed‑point you need
        
        params = {
            'z1': z1, # augmented state vector
            'z2': z2, # augmented state vector
            'A': self.A, # state transition matrix
            'B': self.B, # control input matrix
            'Q': self.Q,  # cost matrix for state
            'R': self.R,  # cost matrix for control inputs
            
            'Pz11': np.random.rand(n, n), # cost-to-go matrix, element 1
            'Pz21': np.random.rand(n**2, n), # cost-to-go matrix, element 2
            'Pz12': np.random.rand(n, n**2), # cost-to-go matrix, element 3
            'Pz22': np.random.rand(n**2, n**2), # cost-to-go matrix, element 4
            'W': np.random.rand(n, n),  # covariance matrix for process noise w 
            # 'V': np.random.rand(n, n),   # covariance matrix for measurement noise v
        }
        
        self.u = newton_method(self.u, params, k=1000)
        
        return 
    
    def update_lqe(self):
        ##### Prediction Step #####
        
        return 
    
    def forward_state(self):
        self.F.forward(u=self.u)
        self.x = self.F.get_current_state()
    
def generate_random_positive_definite_matrix(size, scale=1.0):
    """Generates a random positive definite matrix."""
    A = np.random.randn(size, size)
    return scale * (A.T @ A) + np.eye(size) * 1e-3  # Ensure it's positive definite


