import numpy as np
import scipy.linalg
import scipy.signal
import matplotlib.pyplot as plt

small_value = 1e-6  # Small value to prevent numerical issues

def get_measurement(C, x, M, v):
    quad_term = np.clip(x.T @ M @ x, -1e6, 1e6)  # Limit extreme values
    return C @ x + quad_term * np.ones((M.shape[0], 1)) + v

class LQG_EKF:
    def __init__(self, A_S, B_Si, C, Q, R, W, V, M, H, num_sensors=4):
        # states
        self.H = H # time horizon
        self.m = num_sensors
        self.x = [np.zeros((num_sensors, 1), dtype=np.float64)] # predicted state
        self.x_hat = [np.zeros((num_sensors, 1), dtype=np.float64)] # estimated state
        self.u = [] # control input
        
        # lqe 
        self.C = C.astype(np.float64) # measurement matrix
        self.kalman_gain_list = [None]
        self.P_lqe = [np.eye(self.m, dtype=np.float64) * small_value]  # estimation error covariance matrix 
        self.M = M.astype(np.float64) # matrix in the non-linear measurement function
        
        # lqr
        self.A = A_S.astype(np.float64)
        self.B = B_Si.astype(np.float64)
        self.Q = Q.astype(np.float64)
        self.R = R.astype(np.float64)
        self.control_gain_list = []
        self.P_lqr = [self.Q]
        
        # noise
        self.W = W.astype(np.float64)
        self.V = V.astype(np.float64)
        self.v = [None]
        self.w = [None]
        for _ in range(H):
            self.v.append(np.random.multivariate_normal(np.zeros(self.m), self.V).reshape(-1, 1))
            self.w.append(np.random.multivariate_normal(np.zeros(self.m), self.W).reshape(-1, 1))
    
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
        C_tilde = (self.C + 2 * self.M @ self.x @ np.ones((1, self.m))).squeeze()
        
        for k in range(1, self.H + 1, 1):
            # update state
            self.u.append(self.control_gain_list[k-1] @ self.x_hat[k-1])
            self.x.append(self.A @ self.x[k-1] + self.B @ self.u[k-1] + self.w[k])
            
            # priori estimate 
            x_hat_pri = self.A @ self.x_hat[k-1] + self.B @ self.u[k-1]     
            
            # P_k-1
            p0 = self.P_lqe[k-1]
            
            # kalman gain
            kalman_gain =(p0 @ C_tilde.T @ np.linalg.pinv(C_tilde @ p0 @ C_tilde.T + self.V))
            self.kalman_gain_list.append(kalman_gain)
            
            # measurement
            z = get_measurement(self.C, self.x[k], self.M, self.v[k])
            
            # innovation
            y = z - get_measurement(self.C, x_hat_pri, self.M, np.zeros((self.m, 1)))
            
            # posterior estimate
            x_hat_post = x_hat_pri + kalman_gain @ y
            self.x_hat.append(x_hat_post)
            
            # P_k - Propagation of the estimation error covariance matrix
            p1 = (np.eye(self.m) - kalman_gain @ C_tilde) @ p0
            self.P_lqe.append(p1)
        return
    
    def simulate(self, plot=True):
        self.update_lqr()
        # print("LQR control gain: ", self.control_gain_list)
        
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
    num_sensors = 8
    H = 200  # Time horizon

    # Random system matrices
    A_S = np.eye(num_sensors) + np.random.randn(num_sensors, num_sensors) * 0.1
    B_Si = np.random.randn(num_sensors, num_sensors) * 0.1
    C = np.random.randn(num_sensors, num_sensors)
    Q = np.eye(num_sensors) * 0.1  # State cost
    R = np.eye(num_sensors) * 0.1  # Control cost
    W = np.eye(num_sensors) * 0.05  # Process noise covariance
    V = np.eye(num_sensors) * 0.05  # Measurement noise covariance
    M = np.random.randn(num_sensors, num_sensors) * 0.01  # Nonlinear measurement term

    lqg_system = LQG_EKF(A_S, B_Si, C, Q, R, W, V, M, H, num_sensors)
    lqg_system.simulate(plot=True)