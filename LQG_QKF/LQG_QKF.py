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
from matrix_checker import *
from scipy.linalg import sqrtm

small_value = 1e-6  # Small value to prevent numerical issues

def generate_random_symmetric_matrix(size, scale=1.0):
    """"Generate a random symmetric positive definite matrix."""
    A = np.random.randn(size, size)
    return scale * (A.T @ A) + np.eye(size) * 1e-3  # Ensure it's positive definite

def check_eigenvalues(A_tilde, B_tilde):
    # A_tilde: (20×20), B_tilde: (20×3)
    eigs = eigvals(A_tilde)
    unstable = [lam for lam in eigs if abs(lam) >= 1]
    print('\n')
    print("All eigenvalues of Ã:\n", np.round(eigs,4))
    print("Unstable (|λ|≥1):\n", np.round(unstable,4))

    n_aug = A_tilde.shape[0]
    for lam in unstable:
        M = np.hstack([lam*np.eye(n_aug) - A_tilde, B_tilde])
        r = matrix_rank(M)
        print(f"λ={lam:.4f} → rank([λI-Ã, B̃]) = {r}/{n_aug}")
    return

def check_stability(A, B, Q, R):
    # shape
    # A: (n+n^2, n+n^2), B: (n+n^2, p), Q: (n+n^2, n+n^2), R: (p, p)

    # 2) check structural conditions
    if not is_pos_def(R):
        raise ValueError("R is not positive-definite after regularisation.")
    if not stabilisable(A, B):
        check_eigenvalues(A, B)
        raise ValueError("(A,B) not stabilisable.")
    if not detectable(A, sqrtm(Q)):        # or use C in LQG
        raise ValueError("(A,Q^{1/2}) not detectable.")
    return 0

def finite_horizon_lqr(A, B, Q, R, N=100, Qf=None):
    if Qf is None:
        Qf = Q.copy()
    P = Qf.copy()
    K_list = [None]*N
    # backward recursion
    for k in reversed(range(N)):
        G    = R + B.T @ P @ B
        Kk   = np.linalg.solve(G, B.T @ P @ A)
        P    = Q + A.T @ P @ (A - B @ Kk)
        K_list[k] = Kk
    return P

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
    
    def update_lqr(self, infinite_horizon = False):
        goal_state = self.F.get_current_state() # shape (n,1)
        z_0 = (np.concatenate([goal_state.T, Vec(goal_state @ goal_state.T).T], axis=1)).T
        z_1, z1_1, z2_1 = self.F.aug_state()
        z = z_1 - z_0
        z1 = z[:self.n, :]
        z2 = z[self.n:, :]
        A_tilde = self.F.get_A_tilde() # shape (n+n^2, n+n^2)
        B_tilde = self.F.get_B_tilde() # shape (n+n^2, p)
        
        if infinite_horizon: # infinite horizon LQR -> DOES NOT WORK FOR THIS CASE!
            check_stability(A_tilde, B_tilde, self.Q, self.R)
            from scipy.linalg import solve_discrete_are
            P = solve_discrete_are(A_tilde, B_tilde, self.Q, self.R)   # P is the fixed‑point
        else: # finite horizon LQR
            P = finite_horizon_lqr(A_tilde, B_tilde, self.Q, self.R, N=self.H, Qf=None)
        
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
            'Pz21': P21, # cost-to-go matrix, element 2
            'Pz12': P12, # cost-to-go matrix, element 3
            'Pz22': P22, # cost-to-go matrix, element 4
            'W': self.W,  # covariance matrix for process noise w 
        }
        newton = NewtonSolver(self.n, self.p)
        u_new = newton.newton_method(self.F.get_current_control(), params, epsilon=1e-6, max_iter=1000, verbose=False, plot=False)
        self.F.set_control(u_new) # update control input vector
        return 
    
    def update_lqe(self):
        Phi  = self.F.get_A_tilde()
        Sigma_tilde = self.F.aug_process_noise_covar()
        mu_tilde = self.F.mu_tilde()
    
        # state prediction     Z_{t|t‑1} ,  P⁽ᶻ⁾_{t|t‑1}
        Z_pred = Phi @ self.Z_est + mu_tilde
        P_pred = Phi @ self.Pz_est @ Phi.T + Sigma_tilde

        # measurement prediction Y_{t|t‑1} , innovation cov  M
        measA = self.sensor.get_measA() # shape (m, 1)
        measB_tilde = self.sensor.get_aug_measB() # shape (m, n+n^2)
        Y_pred = measA + measB_tilde @ Z_pred # shape (m, n+n^2)
        measM = self.sensor.get_aug_measB() @ P_pred @ self.sensor.get_aug_measB().T + self.V

        # Kalman gain          Kₜ
        K = P_pred @ self.sensor.get_aug_measB().T @ np.linalg.inv(measM)

        # state update         Z_{t|t} ,  P⁽ᶻ⁾_{t|t}
        z, _, _ = self.F.aug_state()
        Y_meas = self.sensor.aug_measure(z)
        innovation = Y_meas - Y_pred
        self.Z_est = Z_pred + K @ innovation
        
        # Joseph form of the covariance update 
        I = np.eye(P_pred.shape[0])
        Btil = measB_tilde      # your sensor.get_aug_measB()
        V    = self.V

        # Joseph‑form update
        Pj = (I - K @ Btil) @ P_pred @ (I - K @ Btil).T \
            + K @ V @ K.T

        # enforce exact symmetry
        self.Pz_est = 0.5*(Pj + Pj.T)

        # keep the classical x̂ part handy for the LQR
        n = self.n
        self.x_hat = self.Z_est[:n, :]

        return 
    
    def forward_state(self):
        self.F.forward()
        
    def run_sim(self):
        estimate_error_list = []
        for _ in tqdm.tqdm(range(1, self.H + 1, 1)):
            self.update_lqr()
            self.update_lqe()
            self.forward_state()
            
            Z_est = self.Z_est
            x_est = Z_est[:self.n, :]
            err = np.linalg.norm(self.F.get_current_state() - x_est)
            estimate_error_list.append(err)
        return estimate_error_list

def main():
    n = 4
    p = 3
    m = 2
    
    W = generate_random_symmetric_matrix(n, scale=1.0)
    A = np.random.randn(n, n)
    B = np.random.randn(n, p)
    
    F = StateDynamics(n, p , W, A, B)
    C = np.random.randn(m, n)
    
    M = np.random.randn(m, n, n)
    V = generate_random_symmetric_matrix(m, scale=1.0)
    S = sensor(C, M, V)
    
    # Q, R must be symmetric positive definite matrices
    Q = generate_random_symmetric_matrix(n+n**2, scale=1.0)
    R = generate_random_symmetric_matrix(p, scale=1.0)
    lqg_qkf_sys = LQG_QKF(F, S, Q, R, H=50)
    
    err_list = lqg_qkf_sys.run_sim()
    print("Estimate error list: ", err_list)
    plt.plot(err_list, label='Estimate error')
    plt.legend()
    plt.title('Estimate error')
    plt.xlabel('Time step')
    plt.ylabel('Estimate error')
    plt.grid()
    plt.show()
    return

if __name__ == "__main__":
    main()
        



