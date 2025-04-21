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
from matrxi_checker import *

small_value = 1e-6  # Small value to prevent numerical issues

def generate_random_symmetric_matrix(size, scale=1.0):
    """"Generate a random symmetric positive definite matrix."""
    A = np.random.randn(size, size)
    return scale * (A.T @ A) + np.eye(size) * 1e-3  # Ensure it's positive definite

def check_stability(A, B, Q, R):
    # 1) regularise input matrices
    Q  = 0.5*(Q + Q.T)
    R  = 0.5*(R + R.T)
    eps = 1e-8
    np.fill_diagonal(Q, np.maximum(np.diag(Q), eps))
    np.fill_diagonal(R, np.maximum(np.diag(R), eps))

    # 2) check structural conditions
    if not is_pos_def(R):
        raise ValueError("R is not positive-definite after regularisation.")
    if not stabilisable(A, B):
        raise ValueError("(A,B) not stabilisable.")
    if not detectable(A, np.sqrt(Q)):        # or use C in LQG
        raise ValueError("(A,Q^{1/2}) not detectable.")
    return 0

class LQG_QKF:
    def __init__(self, F: StateDynamics, S: sensor,
                 Q, R, H = 50):
        
        # dynamics setting
        self.F = F
        
        # sensor settings
        self.sensor = S
        self.V = S.get_V()
        
        # state settings
        self.A = F.get_A()
        self.B = F.get_B()
        self.n = F.get_state_size()
        self.p = F.get_input_size()
        self.W = F.get_W()
        
        # augmented state settings
        self.Z_est = self.F.mu_tilde() # Z₀ = [x₀, x₀ @ x₀.T] # shape (n+n^2, 1)
        self.Pz_est = self.F.aug_process_noise_covar() # Sigma_tilde, shape (n+n^2, n+n^2)
        

        
        # forward dynamics
        self.F = F
        
        # horizon
        self.H = H
        
        # states
        self.x_hat = np.zeros((self.n, 1)) # estimated state vector
        self.z_hat = np.zeros((self.n + self.n**2, 1)) # estimated augmented state vector
        
        # lqr
        self.Q = Q.astype(np.float64)
        self.R = R.astype(np.float64)
    
    def update_lqr(self):
        goal_state = self.F.get_current_state() # shape (n,1)
        z_0 = (np.concatenate([goal_state.T, Vec(goal_state @ goal_state.T).T], axis=1)).T
        z1_0 = z_0[:self.n, :]
        z2_0 = z_0[self.n:, :]
        z_1, z1_1, z2_1 = self.F.aug_state()
        z = z_1 - z_0
        z1 = z[:self.n, :]
        z2 = z[self.n:, :]
        A_tilde = self.F.get_A_tilde() # shape (n+n^2, n+n^2)
        B_tilde = self.F.get_B_tilde() # shape (n+n^2, p)
        
        if check_stability(A_tilde, B_tilde, self.Q, self.R) != 0:
            raise ValueError("Stability check failed.")
        from scipy.linalg import solve_discrete_are
        P = solve_discrete_are(A_tilde, B_tilde, self.Q, self.R)   # P is the fixed‑point
        n = self.n # state size
        P11 = P[:n,     :n]
        P12 = P[:n,     n:]
        P21 = P[n:,     :n]
        P22 = P[n:,     n:]
        
        params = {
            'z1': z1, # augmented state vector
            'z2': z2, # augmented state vector
            'A': self.A, # state transition matrix
            'B': self.B, # control input matrix
            'Q': self.Q,  # cost matrix for state
            'R': self.R,  # cost matrix for control inputs
            
            'Pz11': P11, # cost-to-go matrix, element 1
            'Pz21': P12, # cost-to-go matrix, element 2
            'Pz12': P21, # cost-to-go matrix, element 3
            'Pz22': P22, # cost-to-go matrix, element 4
            'W': self.W,  # covariance matrix for process noise w 
        }
        newton = NewtonSolver(self.n, self.p)
        self.F.set_control(newton.newton_method(self.F.get_current_control(), params, max_iter=1000)) # update control input vector
        return 
    
    def update_lqe(self):
        """
        One predict-update step of the Quadratic Kalman Filter (QKF).

        Parameters
        ----------
        y_meas : (m,1) ndarray
            The latest sensor measurement  yₜ .
        """
        # ‑‑‑‑‑ ensure all augmented matrices exist

        Phi  = self.F.get_A_tilde()
        Sigma_tilde = self.F.aug_process_noise_covar()
        mu_tilde = self.F.mu_tilde()
    
        # ────────────────────────────────────────────────────────────────
        # 1) state prediction     Z_{t|t‑1} ,  P⁽ᶻ⁾_{t|t‑1}
        # ────────────────────────────────────────────────────────────────
        Z_pred = Phi @ self.Z_est + mu_tilde
        P_pred = Phi @ self.Pz_est @ Phi.T + Sigma_tilde

        # ────────────────────────────────────────────────────────────────
        # 2) measurement prediction Y_{t|t‑1} , innovation cov  M
        # ────────────────────────────────────────────────────────────────
        measA = self.sensor.get_measA() # shape (m, 1)
        measB_tilde = self.sensor.get_aug_measB() # shape (m, n+n^2)
        Y_pred = measA + measB_tilde @ Z_pred # shape (m, n+n^2)
        measM = self.sensor.get_aug_measB() @ P_pred @ self.sensor.get_aug_measB().T + self.V

        # ────────────────────────────────────────────────────────────────
        # 3) Kalman gain          Kₜ
        # ────────────────────────────────────────────────────────────────
        K = P_pred @ self.sensor.get_aug_measB.T @ np.linalg.inv(measM)

        # ────────────────────────────────────────────────────────────────
        # 4) state update         Z_{t|t} ,  P⁽ᶻ⁾_{t|t}
        # ────────────────────────────────────────────────────────────────
        Y_meas = self.sensor.aug_measure(self.F.aug_state())
        innovation = Y_meas - Y_pred
        self.Z_est = Z_pred + K @ innovation
        self.Pz_est = P_pred - K @ measM @ K.T

        # keep the classical x̂ part handy for the LQR
        n = self.n
        self.x_hat = self.Z_est[:n, :]

        return self.Pz_est
    
    def forward_state(self):
        self.F.forward()
        
    def run_sim(self):
        estimate_error_list = []
        for _ in tqdm.tqdm(range(1, self.H + 1, 1)):
            self.update_lqr()
            self.forward_state()
            Pz = self.update_lqe()
            
            err = abs(self.x_hat - self.F.get_current_state())
            estimate_error_list.append(err)
        return estimate_error_list

def main():
    n = 4
    p = 3
    m = 2
    
    W = np.random.randn(n, n)
    A = np.random.randn(n, n)
    B = np.random.randn(n, p)
    
    F = StateDynamics(n, p , W, A, B)
    C = np.random.randn(m, n)
    
    M = np.random.randn(m, n, n)
    V = np.random.randn(m, m)
    S = sensor(C, M, V)
    
    # Q, R must be symmetric positive definite matrices
    Q = generate_random_symmetric_matrix(n+n**2, scale=1.0)
    R = generate_random_symmetric_matrix(p, scale=1.0)
    lqg_qkf_sys = LQG_QKF(F, S, Q, R, H=50)
    
    err_list = lqg_qkf_sys.run_sim()
    plt.plot(err_list)
    plt.title('Estimate error')
    plt.xlabel('Time step')
    plt.ylabel('Estimate error')
    plt.show()
    return

if __name__ == "__main__":
    main()
        



