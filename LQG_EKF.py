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
        
        self.m = A_S.shape[0] # sensor state size
        self.n = A_E.shape[0] # earth state size
        
        x_E_0 = np.zeros((self.n, 1)) # state of the earth
        x_S_0 = np.zeros((self.m, 1)) # state of the sensors
        self.x = [np.vstack((x_E_0, x_S_0))] # true state
        self.x_hat = [np.vstack((x_E_0, x_S_0))] # estimated state
        
        self.u = [] # control input
        
        # lqe 
        self.C = C.astype(np.float64) # measurement matrix
        self.kalman_gain_list = [None]
        self.P_lqe = [np.eye(self.n + self.m) * small_value]  # estimation error covariance matrix 
        self.M = M.astype(np.float64) # matrix in the non-linear measurement function
        
        # lqr
        self.A_E = A_E.astype(np.float64)
        self.A_S = A_S.astype(np.float64)
        self.A = scipy.linalg.block_diag(self.A_E, self.A_S)
        
        self.B_E = np.zeros((self.n, self.n))
        self.B_S = B_Si.astype(np.float64)
        
        self.B = np.linalg.pinv(scipy.linalg.block_diag(self.B_E, self.B_S))
        self.Q = Q.astype(np.float64)
        self.R = R.astype(np.float64)
        self.control_gain_list = []
        self.P_lqr = [self.Q]
        
        # noise
        self.W_E = W_E.astype(np.float64)
        self.W_S = W_S.astype(np.float64)
        self.V = V.astype(np.float64)
        self.w = [None]
        self.v = [None]
        print('Generating noise...')
        for _ in tqdm.tqdm(range(H)):
            w_E = np.random.multivariate_normal(np.zeros(self.n), self.W_E).reshape(-1, 1)
            w_S = np.random.multivariate_normal(np.zeros(self.m), self.W_S).reshape(-1, 1)
            v = np.random.multivariate_normal(np.zeros(self.m + self.n), self.V).reshape(-1, 1)
            self.w.append(np.vstack((w_E, w_S)))
            self.v.append(v)
    
    def get_measurement(self, C, x, v, M):
        assert C is not None, 'Measurement matrix C is required for Kalman filter'
        assert x is not None, 'State x is required for Kalman filter'
        assert v is not None, 'Measurement noise v is required for Kalman filter'
        assert M is not None, 'Matrix M is required for Extended Kalman filter'
        quad_term = np.clip(x.T @ M @ x, -1e6, 1e6)  # Limit extreme values
        return C @ x + quad_term * np.ones((M.shape[0], 1)) + v
    
    def update_lqr(self, infinite_horizon=False):
        print('Calculating LQR...')
        if infinite_horizon: # infinite horizon LQR
            for k in tqdm.tqdm(range(self.H)):
                P_lqr = scipy.linalg.solve_discrete_are(self.A, self.B, self.Q, self.R)  # P is the fixed-point
                self.P_lqr.append(P_lqr)
                feedback_gain = -np.linalg.pinv(self.R + self.B.T @ self.P_lqr[-1] @ self.B) @ self.B.T @ self.P_lqr[-1] @ self.A
                self.control_gain_list.append(feedback_gain)
        else: # finite horizon LQR
            N = 100 # number of steps for finite horizon LQR
            for k in tqdm.tqdm(range(self.H, 0, -1)):
                p_list = []
                p_list.append(self.Q)
                for i in range(1, N):
                    p = self.Q + self.A.T @ p_list[i-1] @ self.A - self.A.T @ p_list[i-1] @ self.B @ np.linalg.pinv(self.R + self.B.T @ p_list[i-1] @ self.B) @ self.B.T @ p_list[i-1] @ self.A
                    p_list.append(p)
                self.P_lqr.append(p_list[-1])
                feedback_gain = -np.linalg.pinv(self.R + self.B.T @ self.P_lqr[-1] @ self.B) @ self.B.T @ self.P_lqr[-1] @ self.A
                self.control_gain_list.append(feedback_gain)
        return
    
    def update_lqe(self): 
        
        if self.filter == 'KF':
            for k in tqdm.tqdm(range(1, self.H + 1, 1)):
                # update state
                self.u.append(self.control_gain_list[k-1] @ self.x_hat[k-1])
                self.x.append(self.A @ self.x[k-1] + self.B @ self.u[k-1] + self.w[k])
                
                # priori estimate 
                x_hat_pri = self.A @ self.x_hat[k-1] + self.B @ self.u[k-1]     
                
                # P_k-1
                p0 = self.P_lqe[k-1]
                
                # kalman gain
                    # K = P- @ C^T @ inv(C @ P- @ C^T + V)
                kalman_gain =(p0 @ self.C.T @ np.linalg.pinv(self.C @ p0 @ self.C.T + self.V))
                self.kalman_gain_list.append(kalman_gain)
                
                # measurement
                z = self.get_measurement(C=self.C, x=self.x[k], v=self.v[k], M=self.M)
                
                # innovation
                y = z - self.get_measurement(C=self.C, x=x_hat_pri, v=np.zeros((self.m + self.n, 1)), M=self.M)
                
                # posterior estimate
                x_hat_post = x_hat_pri + kalman_gain @ y
                self.x_hat.append(x_hat_post)
                
                # P_k - Propagation of the estimation error covariance matrix
                p1 = (np.eye(self.n + self.m) - kalman_gain @ self.C) @ p0
                self.P_lqe.append(p1)
        
        elif self.filter == 'EKF':
            for k in tqdm.tqdm(range(1, self.H + 1, 1)):
                # update state
                self.u.append(self.control_gain_list[k-1] @ self.x_hat[k-1])
                self.x.append(self.A @ self.x[k-1] + self.B @ self.u[k-1] + self.w[k])
                
                # Jacobian of the measurement function  
                C_tilde = (self.C + 2 * self.M @ self.x[k] @ np.ones((1, self.n + self.m))).squeeze()
                
                # priori estimate 
                x_hat_pri = self.A @ self.x_hat[k-1] + self.B @ self.u[k-1]     
                
                # P_k-1
                p0 = self.P_lqe[k-1]
                
                # kalman gain
                    # K = P- @ C^T @ inv(C @ P- @ C^T + V)
                kalman_gain =(p0 @ C_tilde.T @ np.linalg.pinv(C_tilde @ p0 @ C_tilde.T + self.V))
                self.kalman_gain_list.append(kalman_gain)
                
                # measurement
                z = self.get_measurement(C=self.C, x=self.x[k], v=self.v[k], M=self.M)
                
                # innovation
                y = z - self.get_measurement(C=self.C, x=x_hat_pri, v=np.zeros((self.n + self.m, 1)), M=self.M)
                
                # posterior estimate
                x_hat_post = x_hat_pri + kalman_gain @ y
                self.x_hat.append(x_hat_post)
                
                # P_k - Propagation of the estimation error covariance matrix
                p1 = (np.eye(self.n + self.m) - kalman_gain @ C_tilde) @ p0
                self.P_lqe.append(p1)
        return
    
    def simulate(self, plot=True, infinite_horizon=False):
        self.update_lqr(infinite_horizon=infinite_horizon)
        
        self.update_lqe()
        
        if plot:
            self.plot_history()
            
        estimate_error_list = []
        for k in range(1, self.H + 1, 1):
            estimate_error_list.append(np.trace(self.P_lqe[k]))
        return estimate_error_list
    
    def plot_history(self):
        
        estimate_error_list = []
        
        for k in range(1, self.H + 1, 1):
            estimate_error_list.append(np.trace(self.P_lqe[k]))    
        
        plt.plot(estimate_error_list, marker='o', markersize=1)
        plt.xlabel('Time')
        plt.ylabel('Estimation error')
        plt.show()
    
def generate_random_positive_definite_matrix(size, scale=1.0):
    """Generates a random positive definite matrix."""
    A = np.random.randn(size, size)
    return scale * (A.T @ A) + np.eye(size) * 1e-3  # Ensure it's positive definite
    
# Test with random values
if __name__ == "__main__":

    # Define dimensions
    n = 8  # Number of earth states
    m = 4  # Number of sensor states

    # Random system matrices
    np.random.seed(0) # set up random seed for reproducibility
    A_E = np.random.randn(n, n) * 0.1
    A_S = np.random.randn(m, m) * 0.1
    B_Si = np.random.randn(m, m) * 0.1
    C = np.random.randn(m + n, n + m) * 0.1
    M = np.random.randn(m + n, n + m) * 0.1  # Small nonlinearity

    # Cost matrices (should be positive definite)
    Q = generate_random_positive_definite_matrix(n + m, scale=10)  # LQR state cost
    R = generate_random_positive_definite_matrix(n + m, scale=1)   # LQR control cost

    # Covariance matrices (must be positive definite)
    W_E = generate_random_positive_definite_matrix(n, scale=1)  # Process noise (Earth)
    W_S = generate_random_positive_definite_matrix(m, scale=1)  # Process noise (Sensor)
    V = generate_random_positive_definite_matrix(m + n, scale=1)  # Measurement noise

    # Horizon length
    H = 10000

    # Create LQG_EKF instance and simulate
    for inf_hor in [True, False]:
        print(f'Running simulation with infinite horizon: {inf_hor}')
        print('Simulating EKF...')
        label = 'inf_H' if inf_hor else 'f_H'
        lqg_ekf = LQG(A_E, A_S, B_Si, C, Q, R, W_E, W_S, V, M, H, filter='EKF')
        estimate_error_list_ekf = lqg_ekf.simulate(plot=False, infinite_horizon=inf_hor)
        
        print('Simulating KF...')
        lqg_kf = LQG(A_E, A_S, B_Si, C, Q, R, W_E, W_S, V, M, H, filter='KF')
        estimate_error_list_kf = lqg_kf.simulate(plot=False, infinite_horizon=inf_hor)
        
        # Plot estimation error
        plt.plot(estimate_error_list_ekf, label=f'EKF-{label}', marker='o', markersize=1)
        plt.plot(estimate_error_list_kf, label=f'KF-{label}', marker='o', markersize=1)
    plt.xlabel('Time')
    plt.ylabel('Estimation error')
    plt.legend()
    plt.savefig(f'./LQG_EKF_perf.png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
        