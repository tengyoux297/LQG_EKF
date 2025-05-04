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

class LQG:
    def __init__(self, n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, H = 50, filter_type: Literal['qkf', 'ekf', 'kf'] = 'qkf'):
        
        self.filter_type = filter_type
        
        # dynamics setting
        self.F = StateDynamics(n1, n2, p , W, A_E, A_S, B_S)
        n = n1 + n2 # state size
        
        # sensor settings
        self.sensor = sensor(C, M, V)
        self.V = self.sensor.get_V()
        
        # state settings
        self.A = self.F.get_A()
        self.B = self.F.get_B()
        self.n = self.F.get_state_size()
        self.p = self.F.get_input_size()
        self.W = self.F.get_W()
        # augmented state settings
        mu_tilde_u = (np.eye(n+n**2) - self.F.get_A_tilde()).T @ self.F.mu_tilde() # shape (n+n^2, 1)
        self.Z_est = mu_tilde_u # initialize estimated augmented state vector
        I = np.eye(n**2 * (n+1)**2) # shape (n^2(n+1)^2, n^2(n+1)^2)
        Phi_tilde = self.F.get_A_tilde() # shape (n+n^2, n+n^2)
        Sigma_tilde = self.F.aug_process_noise_covar() # shape (n+n^2, n+n^2)
        vec_sigma_tilde_u = (I - np.kron(Phi_tilde, Phi_tilde)) @ Vec(Sigma_tilde) # shape (n^2(n+1)^2, 1)
        self.Pz_est = invVec(vec_sigma_tilde_u) # estimation error covariance matrix
        
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
    
    def get_goal_state(self):   
        goal_state = np.random.randn(self.n, 1)
        return goal_state
    
    def update_lqr_qkf(self, goal_state, infinite_horizon = False):
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
    
    def update_lqr_orig(self, goal_state):
        # LQR update only with original state, no augmented state
        P_lqr = scipy.linalg.solve_discrete_are(self.A, self.B, self.Q[:self.n, :self.n], self.R)  # P is the fixed-point
        feedback_gain = -np.linalg.pinv(self.R + self.B.T @ P_lqr @ self.B) @ self.B.T @ P_lqr @ self.A
        u_new = feedback_gain @ (goal_state - self.x_hat)  # control input
        self.F.set_control(u_new)
        return
        
    def update_lqr(self):
        goal_state = self.get_goal_state()
        if self.filter_type == 'ekf' or self.filter_type == 'kf':
            self.update_lqr_orig(goal_state)
        elif self.filter_type == 'qkf':
            # self.update_lqr_orig(goal_state)
            self.update_lqr_qkf(goal_state, False)
        else:
            raise ValueError("Invalid filter type. Choose 'qkf', 'ekf', or 'kf'.")
        return
        
    
    def update_lqe_qkf(self):
        Phi_tilde  = self.F.get_A_tilde()
        Sigma_tilde = self.F.aug_process_noise_covar()
        mu_tilde = self.F.mu_tilde()
        
        # state prediction     Z_{t|t‑1} ,  P⁽ᶻ⁾_{t|t‑1}
        Z_pred = Phi_tilde @ self.Z_est + mu_tilde
        Pz_pred = Phi_tilde @ self.Pz_est @ Phi_tilde.T + Sigma_tilde

        # measurement prediction Y_{t|t‑1} , innovation cov  M
        measA = self.sensor.get_measA() # shape (m, 1)
        measB_tilde = self.sensor.get_aug_measB() # shape (m, n+n^2)
        Y_pred = measA + measB_tilde @ Z_pred # shape (m, n+n^2)
        M = measB_tilde @ Pz_pred @ measB_tilde.T + self.V

        # Kalman gain          Kₜ
        K = Pz_pred @ measB_tilde.T @ np.linalg.inv(M)

        # state update         Z_{t|t} ,  P⁽ᶻ⁾_{t|t}
        Z, _, _ = self.F.aug_state()
        Y_meas = self.sensor.aug_measure(Z)
        innovation = Y_meas - Y_pred
        self.Z_est = Z_pred + K @ innovation
        Pz_1 = Pz_pred - K @ M @ K.T
        
        self.Pz_est = Pz_1
        self.x_hat = self.Z_est[:self.n, :]
        return K
    
    def update_lqe_ekf(self):
        mu = self.F.B @ self.F.u
        Phi = self.F.A
        Sigma = self.F.W
        
        # state prediction
        X_pred = mu + Phi @ self.x_hat
        P_pred = Phi @ self.P_est @ Phi.T + Sigma
        
        # measurement prediction
        Y_pred = self.sensor.measure_pred(X_pred)
        g = self.sensor.g(X_pred)
        M = g @ P_pred @ g.T + self.sensor.V
        
        # gain
        K = P_pred @ g.T @ np.linalg.inv(M)
        
        # state update
        Y_meas = self.sensor.measure(self.F.get_current_state())
        innov = Y_meas - Y_pred
        self.x_hat = X_pred + K @ innov
        self.P_est = P_pred - K @ M @ K.T
        return K
    
    def update_lqe_kf(self):
        C = self.sensor.C
        # priori estimate 
        x_hat_pri = self.A @ self.x_hat + self.B @ self.F.get_current_control()   
        
        # P_k-1
        p0 = self.A @ self.P_est @ self.A.T + self.W
        
        # kalman gain
            # K = P- @ C^T @ inv(C @ P- @ C^T + V)
        kalman_gain =(p0 @ C.T @ np.linalg.pinv(C @ p0 @ C.T + self.V))
        self.kalman_gain = (kalman_gain)
        
        # measurement
        y = self.sensor.measure(self.F.get_current_state())
        
        # innovation
        innov = y - self.sensor.measure_pred(x_hat_pri)
        
        # posterior estimate
        x_hat_post = x_hat_pri + kalman_gain @ innov
        self.x_hat = (x_hat_post)
        
        # P_k - Propagation of the estimation error covariance matrix
        self.P_est = (np.eye(self.n) - kalman_gain @ C) @ p0
        return kalman_gain
    
    def update_lqe(self):
        if self.filter_type == 'qkf':
            K = self.update_lqe_qkf()
        elif self.filter_type == 'ekf':
            K = self.update_lqe_ekf()
        elif self.filter_type == 'kf':
            K = self.update_lqe_kf()
        else:
            raise ValueError("Invalid filter type. Choose 'qkf', 'ekf', or 'kf'.")
        t = self.F.t
        # print(f'  t={t:4d}', f'‖K_{self.filter_type}‖₂=', np.linalg.norm(K),) if t % 100 == 0 else None
        return

    
    def forward_state(self):
        self.F.forward()
        
    def run_sim(self):
        mse_list = []
        var_list = []
        for _ in tqdm.tqdm(range(1, self.H + 1, 1)):
            # self.update_lqr_orig()
            self.update_lqr()
            self.forward_state()
            self.update_lqe()
            
            # record error
            estimate_error = np.linalg.norm(self.F.get_current_state() - self.x_hat).item()
            mse_list.append(estimate_error)
            
            # record variance
            if self.filter_type == 'qkf':
                var = np.trace(self.Pz_est[:self.n, :self.n])
            elif self.filter_type == 'ekf' or self.filter_type == 'kf':
                var = np.trace(self.P_est)
            var_list.append(var)
            
        return mse_list, var_list

def main():
    n1 = 2
    n2 = 2
    n = n1 + n2 # state size
    p = 3
    m = 2
    
    W = generate_random_symmetric_matrix(n, scale=1e-1)
    A_E = np.random.randn(n1, n1) * 1e-1 # state transition matrix for external state
    A_S = np.random.randn(n2, n2) * 1e-1 # state transition matrix for internal state
    B_S = np.random.randn(n2, p) * 1e-1 # control input matrix for internal state
    
    C = np.random.randn(m, n)
    
    M = []
    for i in range(m):
        M.append(generate_random_symmetric_matrix(n, scale=1e2))
    M = np.array(M)
    
    V = generate_random_symmetric_matrix(m, scale=1e-1)
    
    # Q, R must be symmetric positive definite matrices
    Q = generate_random_symmetric_matrix(n+n**2, scale=1.0)
    # Q = generate_random_symmetric_matrix(n, scale=1.0)
    R = generate_random_symmetric_matrix(p, scale=1.0)
    
    # lqg_kf_sys = LQG(n, p, W, A, B, C, M, V, Q, R, H=1000, filter_type='kf')
    # err_list_kf = lqg_kf_sys.run_sim()
    # plt.plot(err_list_kf, label=f'kf measure error')
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    ax[0].set_title('Estimate error')
    ax[0].set_xlabel('Time step')
    ax[0].set_ylabel('Estimate error')
    ax[1].set_title('Estimate error covariance')
    ax[1].set_xlabel('Time step')   
    ax[1].set_ylabel('Estimate error covariance')
    
    lqg_ekf_sys = LQG(n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, H=1000, filter_type='ekf')
    err_list_ekf, var_list_ekf = lqg_ekf_sys.run_sim()
    ax[0].plot(err_list_ekf, label=f'ekf error')
    ax[1].plot(var_list_ekf, label=f'ekf var')
    lqg_qkf_sys = LQG(n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, H=1000, filter_type='qkf')
    err_list_qkf, var_list_qkf = lqg_qkf_sys.run_sim()
    ax[0].plot(err_list_qkf, label=f'qkf error')
    ax[1].plot(var_list_qkf, label=f'qkf var')
    
    ax[0].legend()
    ax[0].grid()
    ax[1].legend()
    ax[1].grid()
    # print("Estimate error list: ", err_list)
    plt.tight_layout()
    plt.savefig('LQG_QKF/perf_comparison.png')
    plt.show()
    return

if __name__ == "__main__":
    main()
        



