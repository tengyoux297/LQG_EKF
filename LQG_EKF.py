import numpy as np
import scipy.linalg
import scipy.signal
import matplotlib.pyplot as plt

small_value = 1e-6  # Small value to prevent numerical issues

def get_measurement(C, x, M, v):
    quad_term = np.clip(x.T @ M @ x, -1e6, 1e6)  # Limit extreme values
    return C @ x + quad_term * np.ones((M.shape[0], 1)) + v


class LQG_EKF:
    def __init__(self, A_E, A_S, B_Si, C, Q, R, W_E, W_S, V, M, H, num_sensors=4):
        # states
        self.H = H # time horizon
        self.m = num_sensors
        
        x_E_0 = np.zeros((num_sensors, 1), dtype=np.float64) # state of the earth
        x_S_0 = np.zeros((num_sensors, 1), dtype=np.float64) # state of the sensors
        self.x = [np.vstack((x_E_0, x_S_0))] # true state
        self.x_hat = [np.vstack((x_E_0, x_S_0))] # estimated state
        
        self.u = [] # control input
        
        # lqe 
        self.C = C.astype(np.float64) # measurement matrix
        self.kalman_gain_list = [None]
        self.P_lqe = [np.eye(2 * self.m, dtype=np.float64) * small_value]  # estimation error covariance matrix 
        self.M = M.astype(np.float64) # matrix in the non-linear measurement function
        
        # lqr
        self.A_E = A_E.astype(np.float64)
        self.A_S = A_S.astype(np.float64)
        self.A = scipy.linalg.block_diag(self.A_E, self.A_S)
        
        self.B_E = np.zeros((num_sensors, num_sensors), dtype=np.float64)
        self.B_S = B_Si.astype(np.float64)
        self.B = np.vstack((self.B_E, self.B_S))
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
        for _ in range(H):
            w_E = np.random.multivariate_normal(np.zeros(num_sensors), W_E).reshape(-1, 1)
            w_S = np.random.multivariate_normal(np.zeros(num_sensors), W_S).reshape(-1, 1)
            v = np.random.multivariate_normal(np.zeros(2 * num_sensors), V).reshape(-1, 1)
            self.w.append(np.vstack((w_E, w_S)))
            self.v.append(v)
    
    def update_lqr(self):
        p_list = [None] * self.H
        p_list.append(self.Q)
        
        for k in range(self.H, 0, -1):
            p = self.Q + self.A.T @ p_list[k] @ self.A - self.A.T @ p_list[k] @ self.B @ np.linalg.pinv(self.R + self.B.T @ p_list[k] @ self.B) @ self.B.T @ p_list[k] @ self.A
            p_list[k-1] = p
        
        self.P_lqr = p_list
        
        for k in range(self.H):
            feedback_gain = -np.linalg.pinv(self.R + self.B.T @ self.P_lqr[k+1] @ self.B) @ self.B.T @ self.P_lqr[k+1] @ self.A
            self.control_gain_list.append(feedback_gain)
        return
    
    def update_lqe(self): 
        
        for k in range(1, self.H + 1, 1):
            # update state
            self.u.append(self.control_gain_list[k-1] @ self.x_hat[k-1])
            self.x.append(self.A @ self.x[k-1] + self.B @ self.u[k-1] + self.w[k])
            
            # Jacobian of the measurement function  
            C_tilde = (self.C + 2 * self.M @ self.x[k] @ np.ones((1, 2*self.m))).squeeze()
            
            # priori estimate 
            x_hat_pri = self.A @ self.x_hat[k-1] + self.B @ self.u[k-1]     
            
            # P_k-1
            p0 = self.P_lqe[k-1]
            
            # kalman gain
                # K = P- @ C^T @ inv(C @ P- @ C^T + V)
            kalman_gain =(p0 @ C_tilde.T @ np.linalg.pinv(C_tilde @ p0 @ C_tilde.T + self.V))
            self.kalman_gain_list.append(kalman_gain)
            
            # measurement
            z = get_measurement(self.C, self.x[k], self.M, self.v[k])
            
            # innovation
            y = z - get_measurement(self.C, x_hat_pri, self.M, np.zeros((2*self.m, 1)))
            
            # posterior estimate
            x_hat_post = x_hat_pri + kalman_gain @ y
            self.x_hat.append(x_hat_post)
            
            # P_k - Propagation of the estimation error covariance matrix
            p1 = (np.eye(2*self.m) - kalman_gain @ C_tilde) @ p0
            self.P_lqe.append(p1)
        return
    
    def simulate(self, plot=True):
        self.update_lqr()
        
        self.update_lqe()
        
        if plot:
            self.plot_history()
    
    def plot_history(self):
        
        estimate_error_list = []
        
        for k in range(1, self.H + 1, 1):
            estimate_error_list.append(np.trace(self.P_lqe[k]))    
        
        plt.plot(estimate_error_list, marker='o', markersize=1)
        plt.xlabel('Time')
        plt.ylabel('Estimation error')
        plt.show()

# Test with random values
if __name__ == "__main__":
    # Define parameters
    num_sensors = 8
    H = 2000  # Time horizon
    # System Matrices
    A_E = np.eye(num_sensors) + np.random.randn(num_sensors, num_sensors) * 0.05
    A_S = np.eye(num_sensors) + np.random.randn(num_sensors, num_sensors) * 0.05
    B_Si = np.random.randn(num_sensors, num_sensors) * 0.1
    C = np.random.randn(2 * num_sensors, 2 * num_sensors)

    # Cost Matrices
    Q = np.eye(2 * num_sensors) * 0.1
    R = np.eye(num_sensors) * 0.1

    # Noise Covariances
    W_E = np.eye(num_sensors) * 0.01
    W_S = np.eye(num_sensors) * 0.01
    V = np.eye(2 * num_sensors) * 0.01  # Combined measurement noise

    # Quadratic measurement matrix
    M = np.random.randn(2 * num_sensors, 2 * num_sensors) * 0.01

    # Initialize LQG-EKF and run simulation
    lqg_ekf = LQG_EKF(A_E, A_S, B_Si, C, Q, R, W_E, W_S, V, M, H, num_sensors)
    lqg_ekf.simulate()