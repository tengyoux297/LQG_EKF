import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import scipy.linalg
import scipy.signal
import matplotlib.pyplot as plt
from typing import Literal
import tqdm
import numpy as np

# Define the device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

small_value = 1e-6  # Small value to prevent numerical issues

def get_measurement(C=None, x=None, v=None, M=None):
    assert C is not None, 'Measurement matrix C is required for Kalman filter'
    assert x is not None, 'State x is required for Kalman filter'
    assert v is not None, 'Measurement noise v is required for Kalman filter'
    assert M is not None, 'Matrix M is required for Extended Kalman filter'
    quad_term = torch.clamp(x.T @ M @ x, -1e6, 1e6)  # Limit extreme values
    return C @ x + quad_term * torch.ones((M.shape[0], 1), dtype=torch.float64, device=device) + v

class LQG:
    def __init__(self, A_E, A_S, B_Si, C, Q, R, W_E, W_S, V, M, H, filter: Literal['EKF', 'KF'] = 'EKF'):
        # filter type
        self.filter = filter    
        
        # states
        self.H = H # time horizon
        
        self.m = A_S.shape[0] # sensor state size
        self.n = A_E.shape[0] # earth state size
        
        x_E_0 = torch.zeros((self.n, 1), dtype=torch.float64, device=device) # state of the earth
        x_S_0 = torch.zeros((self.m, 1), dtype=torch.float64, device=device) # state of the sensors
        self.x = [torch.vstack((x_E_0, x_S_0))] # true state
        self.x_hat = [torch.vstack((x_E_0, x_S_0))] # estimated state
        
        self.u = [] # control input
        # drone contrtols - control on sensors
        
        # lqe 
        self.C = C.to(dtype=torch.float64, device=device) # measurement matrix
        self.kalman_gain_list = [None]
        self.P_lqe = [torch.eye(self.n + self.m, dtype=torch.float64, device=device) * small_value]  # estimation error covariance matrix 
        self.M = M.to(dtype=torch.float64, device=device) # matrix in the non-linear measurement function
        
        # lqr
        self.A_E = A_E.to(dtype=torch.float64, device=device)
        self.A_S = A_S.to(dtype=torch.float64, device=device)
        self.A = torch.tensor(scipy.linalg.block_diag(self.A_E.cpu(), self.A_S.cpu()), dtype=torch.float64, device=device)
        
        self.B_E = torch.zeros((self.n, self.n), dtype=torch.float64, device=device)
        self.B_S = B_Si.to(dtype=torch.float64, device=device)
        
        self.B = torch.tensor(np.linalg.pinv(scipy.linalg.block_diag(self.B_E.cpu(), self.B_S.cpu())), dtype=torch.float64, device=device)
        self.Q = Q.to(dtype=torch.float64, device=device)
        self.R = R.to(dtype=torch.float64, device=device)
        self.control_gain_list = []
        self.P_lqr = [self.Q]
        
        # noise
        self.W_E = W_E.to(dtype=torch.float64, device=device)
        self.W_S = W_S.to(dtype=torch.float64, device=device)
        self.V = V.to(dtype=torch.float64, device=device)
        self.w = [None]
        self.v = [None]
        print('Generating noise...')
        for _ in tqdm.tqdm(range(H)):
            w_E = torch.tensor(np.random.multivariate_normal(np.zeros(self.n), W_E.cpu().numpy()), dtype=torch.float64, device=device).reshape(-1, 1)
            w_S = torch.tensor(np.random.multivariate_normal(np.zeros(self.m), W_S.cpu().numpy()), dtype=torch.float64, device=device).reshape(-1, 1)
            v = torch.tensor(np.random.multivariate_normal(np.zeros(self.m + self.n), V.cpu().numpy()), dtype=torch.float64, device=device).reshape(-1, 1)
            self.w.append(torch.vstack((w_E, w_S)))
            self.v.append(v)
    
    def update_lqr(self):
        p_list = [None] * self.H
        p_list.append(self.Q)
        print('Calculating LQR...')
        for k in tqdm.tqdm(range(self.H, 0, -1)):
            p = self.Q + self.A.T @ p_list[k] @ self.A - self.A.T @ p_list[k] @ self.B @ torch.linalg.pinv(self.R + self.B.T @ p_list[k] @ self.B) @ self.B.T @ p_list[k] @ self.A
            p_list[k-1] = p
        
        self.P_lqr = p_list
        
        for k in tqdm.tqdm(range(self.H)):
            feedback_gain = -torch.linalg.pinv(self.R + self.B.T @ self.P_lqr[k+1] @ self.B) @ self.B.T @ self.P_lqr[k+1] @ self.A
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
                kalman_gain =(p0 @ self.C.T @ torch.linalg.pinv(self.C @ p0 @ self.C.T + self.V))
                self.kalman_gain_list.append(kalman_gain)
                
                # measurement
                z = get_measurement(C=self.C, x=self.x[k], v=self.v[k], M=self.M)
                
                # innovation
                y = z - get_measurement(C=self.C, x=x_hat_pri, v=torch.zeros((self.m + self.n, 1), dtype=torch.float64, device=device), M=self.M)
                
                # posterior estimate
                x_hat_post = x_hat_pri + kalman_gain @ y
                self.x_hat.append(x_hat_post)
                
                # P_k - Propagation of the estimation error covariance matrix
                p1 = (torch.eye(self.n + self.m, dtype=torch.float64, device=device) - kalman_gain @ self.C) @ p0
                self.P_lqe.append(p1)
        
        elif self.filter == 'EKF':
            for k in tqdm.tqdm(range(1, self.H + 1, 1)):
                # update state
                self.u.append(self.control_gain_list[k-1] @ self.x_hat[k-1])
                self.x.append(self.A @ self.x[k-1] + self.B @ self.u[k-1] + self.w[k])
                
                # Jacobian of the measurement function  
                C_tilde = (self.C + 2 * self.M @ self.x[k] @ torch.ones((1, self.n + self.m), dtype=torch.float64, device=device)).squeeze()
                
                # priori estimate 
                x_hat_pri = self.A @ self.x_hat[k-1] + self.B @ self.u[k-1]     
                
                # P_k-1
                p0 = self.P_lqe[k-1]
                
                # kalman gain
                    # K = P- @ C^T @ inv(C @ P- @ C^T + V)
                kalman_gain =(p0 @ C_tilde.T @ torch.linalg.pinv(C_tilde @ p0 @ C_tilde.T + self.V))
                self.kalman_gain_list.append(kalman_gain)
                
                # measurement
                z = get_measurement(C=self.C, x=self.x[k], v=self.v[k], M=self.M)
                
                # innovation
                y = z - get_measurement(C=self.C, x=x_hat_pri, v=torch.zeros((self.n + self.m, 1), dtype=torch.float64, device=device), M=self.M)
                
                # posterior estimate
                x_hat_post = x_hat_pri + kalman_gain @ y
                self.x_hat.append(x_hat_post)
                
                # P_k - Propagation of the estimation error covariance matrix
                p1 = (torch.eye(self.n + self.m, dtype=torch.float64, device=device) - kalman_gain @ C_tilde) @ p0
                self.P_lqe.append(p1)
        return
    
    def simulate(self, plot=True):
        self.update_lqr()
        
        self.update_lqe()
        
        if plot:
            self.plot_history()
            
        estimate_error_list = []
        for k in range(1, self.H + 1, 1):
            estimate_error_list.append(torch.trace(self.P_lqe[k]).item())
        return estimate_error_list
    
    def plot_history(self):
        
        estimate_error_list = []
        
        for k in range(1, self.H + 1, 1):
            estimate_error_list.append(torch.trace(self.P_lqe[k]).item())    
        
        plt.plot(estimate_error_list, marker='o', markersize=1)
        plt.xlabel('Time')
        plt.ylabel('Estimation error')
        plt.show()
    
def generate_random_positive_definite_matrix(size, scale=1.0):
    """Generates a random positive definite matrix."""
    A = torch.randn(size, size, dtype=torch.float64, device=device)
    return scale * (A.T @ A) + torch.eye(size, dtype=torch.float64, device=device) * 1e-3  # Ensure it's positive definite
    
# Test with random values
if __name__ == "__main__":

    # Define dimensions
    n = 8  # Number of earth states
    m = 4  # Number of sensor states

    # Random system matrices
    A_E = torch.randn(n, n, dtype=torch.float64, device=device) * 0.1
    A_S = torch.randn(m, m, dtype=torch.float64, device=device) * 0.1
    B_Si = torch.randn(m, m, dtype=torch.float64, device=device) * 0.1
    C = torch.randn(m + n, n + m, dtype=torch.float64, device=device) * 0.1
    M = torch.randn(m + n, n + m, dtype=torch.float64, device=device) * 0.1  # Small nonlinearity

    # Cost matrices (should be positive definite)
    Q = generate_random_positive_definite_matrix(n + m, scale=10)  # LQR state cost
    R = generate_random_positive_definite_matrix(n + m, scale=1)   # LQR control cost

    # Covariance matrices (must be positive definite)
    W_E = generate_random_positive_definite_matrix(n, scale=1)  # Process noise (Earth)
    W_S = generate_random_positive_definite_matrix(m, scale=1)  # Process noise (Sensor)
    V = generate_random_positive_definite_matrix(m + n, scale=1)  # Measurement noise

    # Horizon length
    H = 1000

    # Create LQG_EKF instance and simulate
    print('Simulating EKF...')
    lqg_ekf = LQG(A_E, A_S, B_Si, C, Q, R, W_E, W_S, V, M, H, filter='EKF')
    estimate_error_list_ekf = lqg_ekf.simulate(plot=False)
    
    print('Simulating KF...')
    lqg_kf = LQG(A_E, A_S, B_Si, C, Q, R, W_E, W_S, V, M, H, filter='KF')
    estimate_error_list_kf = lqg_kf.simulate(plot=False)
    
    # Plot estimation error
    plt.plot(estimate_error_list_ekf, label='EKF', marker='o', markersize=1)
    plt.plot(estimate_error_list_kf, label='KF', marker='o', markersize=1)
    plt.xlabel('Time')
    plt.ylabel('Estimation error')
    plt.legend()
    plt.show()