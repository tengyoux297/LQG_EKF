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
    # backward recursion
    for k in reversed(range(N)):
        P = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.pinv(R + B.T @ P @ B) @ B.T @ P @ A
    return P

class LQG_QKF:
    def __init__(self, F: StateDynamics, S: sensor, Q, R, H = 50):
        
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
        
        # lqe
        self.P_est = np.eye(self.n) * small_value  # estimation error covariance matrix 
    
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
    
    def update_lqr_ekf(self):
        # LQR update only with original state, no augmented state
        P_lqr = scipy.linalg.solve_discrete_are(self.A, self.B, self.Q[:self.n, :self.n], self.R)  # P is the fixed-point
        feedback_gain = -np.linalg.pinv(self.R + self.B.T @ P_lqr @ self.B) @ self.B.T @ P_lqr @ self.A
        u_new = feedback_gain @ (self.F.get_current_state() - self.x_hat)  # control input
        self.F.set_control(u_new)
        return
        
    
    def update_lqe_qkf(self):
        Phi  = self.F.get_A_tilde()
        Sigma_tilde = self.F.aug_process_noise_covar()
        mu_tilde = self.F.mu_tilde()
        
        # state prediction     Z_{t|t‑1} ,  P⁽ᶻ⁾_{t|t‑1}
        Z_pred = Phi @ self.Z_est + mu_tilde
        Pz_pred = Phi @ self.Pz_est @ Phi.T + Sigma_tilde

        # measurement prediction Y_{t|t‑1} , innovation cov  M
        measA = self.sensor.get_measA() # shape (m, 1)
        measB_tilde = self.sensor.get_aug_measB() # shape (m, n+n^2)
        Y_pred = measA + measB_tilde @ Z_pred # shape (m, n+n^2)
        estM = self.sensor.get_aug_measB() @ Pz_pred @ measB_tilde.T + self.V

        # Kalman gain          Kₜ
        K = Pz_pred @ self.sensor.get_aug_measB().T @ np.linalg.inv(estM)

        # state update         Z_{t|t} ,  P⁽ᶻ⁾_{t|t}
        z, _, _ = self.F.aug_state()
        Y_meas = self.sensor.aug_measure(z)
        innovation = Y_meas - Y_pred
        self.Z_est = Z_pred + K @ innovation
        
        Pz_1 = Pz_pred - K @ estM @ K.T
        self.Pz_est = Pz_1
        # keep the classical x̂ part handy for the LQR
        n = self.n
        self.x_hat = self.Z_est[:n, :]
        return 
    
    def update_lqe_ekf(self):
        # Jacobian of the measurement function
        m = self.sensor.C.shape[0]
        temp_term = [None] * m
        for i in range(m):
            temp_term[i] = 2 * self.sensor.M[i] @ self.F.get_current_state()
        temp_term = np.array(temp_term)
        C_tilde = self.sensor.C + temp_term.squeeze()
        
        # priori estimate 
        x_hat_pri = self.A @ self.x_hat + self.B @ self.F.get_current_control()   
        
        # P_k-1
        p0 = self.A @ self.P_est @ self.A.T + self.W
        
        # kalman gain
            # K = P- @ C^T @ inv(C @ P- @ C^T + V)
        kalman_gain =(p0 @ C_tilde.T @ np.linalg.pinv(C_tilde @ p0 @ C_tilde.T + self.V))
        self.kalman_gain = (kalman_gain)
        
        # measurement
        z = self.sensor.measure(self.F.get_current_state())
        
        # innovation
        innov = z - self.sensor.measure_pred(x_hat_pri)
        
        # posterior estimate
        x_hat_post = x_hat_pri + kalman_gain @ innov
        self.x_hat = (x_hat_post)
        
        # P_k - Propagation of the estimation error covariance matrix
        p1 = (np.eye(self.n) - kalman_gain @ C_tilde) @ p0
        self.P_est = (p1)
        return
    
    
    def forward_state(self):
        self.F.forward()
        
    def run_sim_qkf(self):
        estimate_error_list = []
        for _ in tqdm.tqdm(range(1, self.H + 1, 1)):
            # self.update_lqr_orig()
            self.update_lqr()
            self.forward_state()
            self.update_lqe_qkf()
            
            # print('u:', self.F.get_current_control())
            estimate_error_list.append(np.trace(self.Pz_est[:self.n, :self.n]))
        return estimate_error_list
    
    def run_sim_ekf(self):
        estimate_error_list = []
        for _ in tqdm.tqdm(range(1, self.H + 1, 1)):
            # self.update_lqr_orig()
            self.update_lqr_ekf()
            self.forward_state()
            self.update_lqe_ekf()
            
            estimate_error_list.append(np.trace(self.P_est))
        return estimate_error_list

def main():
    n = 4
    p = 3
    m = 2
    
    W = generate_random_symmetric_matrix(n, scale=1e-2)
    A = np.random.randn(n, n) * 0.8 * 1e-2
    B = np.random.randn(n, p) * 0.5 * 1e-2
    
    
    C = np.random.randn(m, n)
    
    M = np.random.randn(m, n, n) * 1e-2
    V = generate_random_symmetric_matrix(m, scale=1e-2)
    
    # Q, R must be symmetric positive definite matrices
    Q = generate_random_symmetric_matrix(n+n**2, scale=1.0)
    # Q = generate_random_symmetric_matrix(n, scale=1.0)
    R = generate_random_symmetric_matrix(p, scale=1.0)
    F1 = StateDynamics(n, p , W, A, B)
    S1 = sensor(C, M, V)
    lqg_ekf_sys = LQG_QKF(F1, S1, Q, R, H=1000)
    err_list_ekf = lqg_ekf_sys.run_sim_ekf()
    F2 = StateDynamics(n, p , W, A, B)
    S2 = sensor(C, M, V)
    lqg_qkf_sys = LQG_QKF(F2, S2, Q, R, H=1000)
    err_list_qkf = lqg_qkf_sys.run_sim_qkf()
    # print("Estimate error list: ", err_list)
    plt.plot(err_list_ekf, label=f'ekf measure error')
    plt.plot(err_list_qkf, label=f'qkf measure error')
    plt.legend()
    plt.title('Estimate error')
    plt.xlabel('Time step')
    plt.ylabel('Estimate error')
    plt.grid()
    plt.savefig('LQG_QKF/qkf_vs_ekf.png')
    plt.show()
    return

if __name__ == "__main__":
    main()
        



