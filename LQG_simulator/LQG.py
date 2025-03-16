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
    def __init__(self, x_0, x_hat_0, F, A, B,  sensor, Q, R, W, V, dt = 0.1, filter: Literal['EKF', 'KF'] = 'EKF'):
        # filter type
        self.filter = filter    
        self.sensor = sensor
        # forward dynamics
        self.F = F
        self.dt = dt
        
        # states
        self.m = A.shape[0] # state size
        
        self.x = x_0 # true state
        self.x_hat = x_hat_0 # estimated state
        
        self.u = None # control input
        
        # lqe 
        self.kalman_gain = None # kalman gain
        self.P_lqe = np.eye(self.m) * small_value  # estimation error covariance matrix
        
        # lqr
        self.A = A.astype(np.float64)
        self.B = B.astype(np.float64)
        self.Q = Q.astype(np.float64)
        self.R = R.astype(np.float64)
        self.P_lqr = self.Q
        
        # noise
        self.W = W.astype(np.float64)
        self.V = V.astype(np.float64)
        self.w = None
        self.v = None

    def update_noise(self):
        self.w = np.random.multivariate_normal(np.zeros(self.W.shape[0]), self.W)
        self.v = np.random.multivariate_normal(np.zeros(self.V.shape[0]), self.V)
    
    def update_lqr(self, goal_state):
        # lqr horizon
        N = 50
        
        self.A, self.B = self.F.linearize(self.x_hat, self.dt)
        
        p_list = [None] * N
        k_list = [None] * N
        u_list = [None] * N   
        
        x_error = self.x - goal_state[0:3]
        # update cost-to-go matrix
        p_list.append(self.Q)
        for k in (range(N, 0, -1)):
            p = self.Q + self.A.T @ p_list[k] @ self.A - self.A.T @ p_list[k] @ self.B @ np.linalg.pinv(self.R + self.B.T @ p_list[k] @ self.B) @ self.B.T @ p_list[k] @ self.A
            p_list[k-1] = p
        
        # update control gain
        for k in (range(N)):
            feedback_gain = -np.linalg.pinv(self.R + self.B.T @ self.P_lqr @ self.B) @ self.B.T @ self.P_lqr @ self.A
            k_list[k] = feedback_gain
        
        # update control input
        for k in (range(N)):
            u_list[k] = k_list[k] @ x_error
        
        # get optimal control input
        self.u = u_list[N-1]
        
        return 
    
    def update_lqe(self):
        
        ##### Prediction Step #####
    
        # Predict the state estimate based on the previous state and the 
        # control input
        # x_predicted is a 3x1 vector (X_t+1, Y_t+1, THETA_t+1)
        x_predicted = self.F.forward(self.x_hat, self.u, self.v, self.dt)
    
        # Calculate the A and B matrices    
        A, B = self.F.linearize(x=self.x_hat)
        
        # Predict the covariance estimate based on the 
        # previous covariance and some noise
        # A and V are 3x3 matrices
        # sigma is state covariance matrix
        sigma_3by3 = None
        if (self.P_lqe.size == 3):
            sigma_3by3 = self.P_lqe * np.eye(3)
        else:
            sigma_3by3 = self.P_lqe
        
        sigma_0 = A @ sigma_3by3 @ A.T + self.V
    
        ##### Correction Step #####  
    
        # Get H, the 4x3 Jacobian matrix for the sensor
        C_tilde = self.sensor.jacobian(self.x_hat) 
    
        # Calculate the observation model   
        # y_predicted is a 4x1 vector
        y_predicted = self.sensor.measure(x_predicted, self.w)
    
        # Measurement residual (delta_y is a 4x1 vector)
        y = self.sensor.measure(self.x, self.w)
        delta_y = y - y_predicted
    
        # Compute the Kalman gain
        #   K = P- @ C^T @ inv(C @ P- @ C^T + V)
        # The Kalman gain indicates how certain you are about
        # the observations with respect to the motion
        K = sigma_0 @ C_tilde.T @ np.linalg.pinv(C_tilde @ sigma_0 @ C_tilde.T + self.W)
    
        # Update the state estimate
        # 3x1 + (3x4 @ 4x1 -> 3x1)
        x_hat_2 = x_predicted + (K @ delta_y)
        
        # Update the covariance estimate
        sigma_1 = sigma_0 - (K @ C_tilde @ sigma_0)
        
        self.x_hat = x_hat_2
        self.P_lqe = sigma_1
        return 
    
    def forward_state(self):
        self.x = self.F.forward(self.x, self.u, self.v, self.dt)
    
def generate_random_positive_definite_matrix(size, scale=1.0):
    """Generates a random positive definite matrix."""
    A = np.random.randn(size, size)
    return scale * (A.T @ A) + np.eye(size) * 1e-3  # Ensure it's positive definite