import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import scipy.linalg
import scipy.signal
import matplotlib.pyplot as plt
from typing import Literal
import tqdm

small_value = 1e-6  # Small value to prevent numerical issues

class LQG:
    def __init__(self, A_E, A_S, B_Si, C, Q, R, W_E, W_S, V, M, H, filter: Literal['EKF', 'KF'] = 'EKF'):
        # filter type
        self.filter = filter    
        
        # states
        self.H = H # time horizon
        
        self.n1 = A_E.shape[0] # earth state size
        self.n2 = A_S.shape[0] # sensor state size
        self.m = M.shape[0] # measurement size
        self.p = Q.shape[0] # control input size
        
        x_E_0 = np.zeros((self.n1, 1)) # state of the earth
        x_S_0 = np.zeros((self.n2, 1)) # state of the sensors
        self.x = np.vstack((x_E_0, x_S_0)) # true state
        self.x_hat = np.vstack((x_E_0, x_S_0)) # estimated state
        
        self.u = None # control input
        
        # lqe 
        self.C = C.astype(np.float64) # measurement matrix
        self.kalman_gain = None
        self.P_lqe = np.eye(self.n1 + self.n2) * small_value  # estimation error covariance matrix 
        self.M = M.astype(np.float64) # matrix in the non-linear measurement function
        
        # lqr
        self.A_E = A_E.astype(np.float64)
        self.A_S = A_S.astype(np.float64)
        self.A = scipy.linalg.block_diag(self.A_E, self.A_S)
        
        self.B_E = np.zeros((self.n1, 0)) # no control input for earth states
        self.B_S = B_Si.astype(np.float64)
        self.B = scipy.linalg.block_diag(self.B_E, self.B_S)
        
        self.Q = Q.astype(np.float64)
        self.R = R.astype(np.float64)
        self.control_gain = None
        self.P_lqr = self.Q.copy()
        
        # noise
        self.W_E = W_E.astype(np.float64)
        self.W_S = W_S.astype(np.float64)
        self.W = scipy.linalg.block_diag(W_E,W_S)
        self.V = V.astype(np.float64)
        self.generate_noise_w()
        self.generate_noise_v()
    
    def generate_noise_w(self):
        
        w_E = np.random.multivariate_normal(np.zeros(self.n1), self.W_E).reshape(-1, 1)
        w_S = np.random.multivariate_normal(np.zeros(self.n2), self.W_S).reshape(-1, 1)
        self.w = np.vstack((w_E, w_S))
        
    def generate_noise_v(self):
        v = np.random.multivariate_normal(np.zeros(m), self.V).reshape(-1, 1)
        self.v = v
        
    def get_measurement(self, C, x, v, M):
        assert C is not None, 'Measurement matrix C is required for Kalman filter'
        assert x is not None, 'State x is required for Kalman filter'
        assert v is not None, 'Measurement noise v is required for Kalman filter'
        assert M is not None, 'Matrix M is required for Extended Kalman filter'
        self.generate_noise_v()
        quad_term = np.zeros((self.m, 1))
        for i in range(M.shape[0]):
            quad_term += (x.T @ M[i] @ x).item()
            e = np.zeros((self.m, 1))
            e[i] = 1
            quad_term += e @ x.T @ M[i] @ x
        return C @ x + quad_term + v
    
    def update_lqr(self, infinite_horizon=False):
        if infinite_horizon: # infinite horizon LQR
            self.P_lqr = scipy.linalg.solve_discrete_are(self.A, self.B, self.Q, self.R)  # P is the fixed-point
            feedback_gain = -np.linalg.pinv(self.R + self.B.T @ self.P_lqr @ self.B) @ self.B.T @ self.P_lqr @ self.A
            self.control_gain = (feedback_gain)
        else: # finite horizon LQR
            N = 100 # number of steps for finite horizon LQR
            p_list = [self.P_lqr]
            for i in range(1, N):
                p = self.Q + self.A.T @ p_list[i-1] @ self.A - self.A.T @ p_list[i-1] @ self.B @ np.linalg.pinv(self.R + self.B.T @ p_list[i-1] @ self.B) @ self.B.T @ p_list[i-1] @ self.A
                p_list.append(p)
            self.P_lqr = p_list[-1]
            self.control_gain = -np.linalg.pinv(self.R + self.B.T @ self.P_lqr @ self.B) @ self.B.T @ self.P_lqr @ self.A
        return
    
    def forward_state(self):
        # update state
        self.generate_noise_w()
        self.generate_noise_v()
        # self.u = (self.control_gain @ (self.x_hat - self.x))
        self.u = (self.control_gain @ self.x_hat)  # control input
        self.x = (self.A @ self.x + self.B @ self.u + self.w)
    
    def update_lqe(self): 
        if self.filter == 'KF':
            # priori estimate 
            x_hat_pri = self.A @ self.x_hat + self.B @ self.u    
            
            # P_k-1
            p0 = self.A @ self.P_lqe @ self.A.T + self.W
            
            # kalman gain
                # K = P- @ C^T @ inv(C @ P- @ C^T + V)
            kalman_gain =(p0 @ self.C.T @ np.linalg.pinv(self.C @ p0 @ self.C.T + self.V))
            self.kalman_gain = (kalman_gain)
            
            # measurement
            z = self.get_measurement(C=self.C, x=self.x, v=self.v, M=self.M)
            
            # innovation
            innov = z - self.get_measurement(C=self.C, x=x_hat_pri, v=np.zeros((self.m, 1)), M=self.M)
            
            # posterior estimate
            x_hat_post = x_hat_pri + kalman_gain @ innov
            self.x_hat = (x_hat_post)
            
            # P_k - Propagation of the estimation error covariance matrix
            p1 = (np.eye(self.n1 + self.n2) - kalman_gain @ self.C) @ p0
            self.P_lqe = (p1)
        
        elif self.filter == 'EKF':
            # Jacobian of the measurement function
            temp_term = [None] * self.m
            for i in range(m):
                temp_term[i] = 2 * self.M[i] @ self.x
            temp_term = np.array(temp_term)
            C_tilde = self.C + temp_term.squeeze()
            
            # priori estimate 
            x_hat_pri = self.A @ self.x_hat + self.B @ self.u   
            
            # P_k-1
            p0 = self.A @ self.P_lqe @ self.A.T + self.W
            
            # kalman gain
                # K = P- @ C^T @ inv(C @ P- @ C^T + V)
            kalman_gain =(p0 @ C_tilde.T @ np.linalg.pinv(C_tilde @ p0 @ C_tilde.T + self.V))
            self.kalman_gain = (kalman_gain)
            
            # measurement
            z = self.get_measurement(C=self.C, x=self.x, v=self.v, M=self.M)
            
            # innovation
            innov = z - self.get_measurement(C=self.C, x=x_hat_pri, v=np.zeros((self.m, 1)), M=self.M)
            
            # posterior estimate
            x_hat_post = x_hat_pri + kalman_gain @ innov
            self.x_hat = (x_hat_post)
            
            # P_k - Propagation of the estimation error covariance matrix
            p1 = (np.eye(self.n1 + self.n2) - kalman_gain @ C_tilde) @ p0
            self.P_lqe = (p1)
        return
    
    def simulate(self, infinite_horizon=False):
        estimate_error_list = []
        cost_to_go_list = []
        for k in tqdm.tqdm(range(1, self.H + 1, 1)):
            self.update_lqr(infinite_horizon=infinite_horizon)
            self.forward_state()
            self.update_lqe()
        
            estimate_error_list.append(np.trace(self.P_lqe))
            
            e = self.x_hat - self.x
            Sigma = e @ e.T
            cost_to_go_list.append((self.x_hat.T @ self.P_lqr @ self.x_hat + np.trace(self.P_lqr @ Sigma)).item())
        return estimate_error_list, cost_to_go_list
    
def generate_random_positive_definite_matrix(size, scale=1.0):
    """Generates a random positive definite matrix."""
    A = np.random.randn(size, size)
    return scale * (A.T @ A) + np.eye(size) * 1e-3  # Ensure it's positive definite
    
# Test with random values
if __name__ == "__main__":

    # Define dimensions
    n1 = 8  # Number of earth states
    n2 = 4  # Number of sensor states
    p = 3  # Number of control inputs
    m = 2  # Number of measurements

    # Random system matrices
    np.random.seed(0) # set up random seed for reproducibility
    A_E = np.random.randn(n1, n1) * 0.1
    A_S = np.random.randn(n2, n2) * 0.1
    B_Si = np.random.randn(n2, p) * 0.1
    C = np.random.randn(m, n1 + n2) * 0.1
    M = []
    for i in range(m):
        M.append(np.random.randn(n1 + n2, n1 + n2) * 1e-3)
    M = np.array(M)  # Convert list of matrices to a 3D numpy array

    # Cost matrices (should be positive definite)
    Q = generate_random_positive_definite_matrix(n1 + n2, scale=1)  # LQR state cost
    R = generate_random_positive_definite_matrix(p, scale=1)   # LQR control cost

    # Covariance matrices (must be positive definite)
    W_E = generate_random_positive_definite_matrix(n1, scale=1e-3)  # Process noise (Earth)
    W_S = generate_random_positive_definite_matrix(n2, scale=1e-3)  # Process noise (Sensor)
    V = generate_random_positive_definite_matrix(m, scale=1e-3)  # Measurement noise

    # Horizon length
    H = 1000

    # Create LQG_EKF instance and simulate
    fig, ax = plt.subplots(2,1,figsize=(10, 6))
    for inf_hor in [True, False]:
        print('Running simulation with', 'infinite' if inf_hor else 'finite', 'horizon')
        print('     Simulating EKF...')
        label = 'inf_H' if inf_hor else 'f_H'
        lqg_ekf = LQG(A_E, A_S, B_Si, C, Q, R, W_E, W_S, V, M, H, filter='EKF')
        estimate_error_list_ekf, cost_to_go_list_ekf = lqg_ekf.simulate(infinite_horizon=inf_hor)
        
        print('     Simulating KF...')
        lqg_kf = LQG(A_E, A_S, B_Si, C, Q, R, W_E, W_S, V, M, H, filter='KF')
        estimate_error_list_kf, cost_to_go_list_kf = lqg_kf.simulate(infinite_horizon=inf_hor)
        
        # Plot estimation error
        ax[0].plot(estimate_error_list_ekf, label=f'EKF-{label}', marker='o', markersize=1)
        ax[0].plot(estimate_error_list_kf, label=f'KF-{label}', marker='o', markersize=1)
        #  Plot cost-to-go
        ax[1].plot(cost_to_go_list_ekf, label=f'EKF-{label}', marker='o', markersize=1)
        ax[1].plot(cost_to_go_list_kf, label=f'KF-{label}', marker='o', markersize=1)
        
    ax[0].set_title('Estimation Error')
    ax[0].set_xlabel('Time step')
    ax[0].set_ylabel('Estimation Error')
    ax[0].grid()    
    ax[0].legend()
    
    ax[1].set_title('Cost-to-Go')
    ax[1].set_xlabel('Time step')
    ax[1].set_ylabel('Cost-to-Go')
    ax[1].grid()    
    ax[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'./LQG_EKF_perf.png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
        