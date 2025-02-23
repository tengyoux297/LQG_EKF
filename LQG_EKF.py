import numpy as np
import scipy.linalg
import scipy.signal
import matplotlib.pyplot as plt

small_value = 1e-6

def get_measurement(C, x, M, v):
    return C @ x + (x.T @ M @ x) * np.ones((M.shape[0],1)) + v

class LQG_EKF:
    # here:
    # x includes only x_S, but no x_E
    # u includes only u_Si
    # A includes only A_S
    # B includes only B_Si
    def __init__(self, A_E, A_S, B_Si, C, Q, R, W, V, M, num_sensors=4):
        self.m = num_sensors # number of sensors
        # Define system matrices
        # self.A = scipy.linalg.block_diag(A_E, A_S) # Block diagonal matrix
        # self.B = np.vstack([np.zeros((2, 1)), B_Si]) # Stack B_Si below zeros((2, 1)), because B only applies to A_S
        self.A = A_S
        self.B = B_Si
        self.C = C # Measurement matrix
        self.Q = Q # State cost weight in LQR
        self.R = R # Control cost weight in LQR
        
        self.W = W # Process noise covariance
        self.V = V # Measurement noise covariance
        self.v = np.zeros((num_sensors, 1)) # Measurement noise
        self.w = np.zeros((num_sensors, 1)) # Process noise
        
        # covariance matrices
        self.P_lqe = np.zeros((num_sensors, num_sensors))
        self.P_lqr = self.Q  
        
        # Initialize state variables
        self.x = np.zeros((num_sensors, 1))  # Initial state
        
        self.x_hat = np.zeros((num_sensors, 1))  # Initial estimated state
        self.u = np.zeros((num_sensors, 1))  # Initial control input
        
        self.M = M # matrix involved in EKF measurement update
        assert M.shape[0] == num_sensors and M.shape[1] == num_sensors # Ensure M is of the right shape
        self.t = 0 # time step
        
        # history data
        self.cov_hist = []
        self.cost_hist = []
        
    
    def update_lqe(self):
        '''Update the Kalman filter gain at current time step, and return the error'''
        # Compute Jacobian H_k
        C_tilde = self.C + 2 * self.M @ self.x @ np.ones((1, self.m))
        # Prior covariance matrix at current time step:
        #   P- = A P A^T + W
        self.P_lqe = self.A @ self.P_lqe @ self.A.T + self.W
        
        # Kalman gain at current time step: 
        #   K = P- C_T (R + C P- C_T)^-1
        self.L = self.P_lqe @ C_tilde.T @ np.linalg.inv(C_tilde @ self.P_lqe @ C_tilde.T + self.V)
        
        # Posterior state estimate at current time step:
        #   x+ = x- + K (y - C x-)
        y = get_measurement(self.C, self.x, self.M, self.v) # y(t) = C x(t) + M + v
        y_hat = get_measurement(self.C, self.x_hat, self.M, np.zeros((self.m, 1)))
        
        self.x_hat = self.x_hat + self.L @ (y - y_hat)
        
        # Posterior state estimate at current time step:
        #  P+ = (I - K C) P-
        self.P_lqe = (np.eye(self.m) - self.L @ C_tilde) @ self.P_lqe 
        return
    
    def update_lqr(self):
        '''Update the LQR gain at current time step, and return the cost-to-go matrix'''
        
        # Prior covariance matrix at current time step:
        #   P = A^T P A - (A^T P B) (R + B^T P B)^(-1) (B^T P A) + Q
        self.P_lqr = self.A.T @ self.P_lqr @ self.A - (self.A.T @ self.P_lqr @ self.B) @ \
                    np.linalg.inv(self.R + self.B.T @ self.P_lqr @ self.B) @ (self.B.T @ self.P_lqr @ self.A) + self.Q
                    
        # control gain at current time step:
        #   K = (R + B^T P B)^(-1) (B^T P A)
        self.K = np.linalg.inv(self.R + self.B.T @ self.P_lqr @ self.B) @ (self.B.T @ self.P_lqr @ self.A)  # LQR gain
        return 
    
    def step(self):
        # Process dynamics
        self.w = np.random.multivariate_normal(mean=np.zeros(self.m), cov=self.W).reshape(-1, 1)
        self.v = np.random.multivariate_normal(mean=np.zeros(self.m), cov=self.V).reshape(-1, 1)
        
        # State prediction
        self.x = self.A @ self.x + self.B @ self.u + self.w  
        
        # Kalman filter update (LQE)
        self.update_lqe()
        estimate_error = np.trace(self.P_lqe)  # Trace of estimation error covariance
        
        # LQR update
        self.update_lqr()
        
        # Compute control cost correctly
        #   deviation-cost-to-go + immediate control effort cost
        cost_to_go = (self.x_hat.T @ self.P_lqr @ self.x_hat + self.u.T @ self.R @ self.u) # xPx + uRu
        control_cost = cost_to_go.item()

        # Compute control input using LQR
        self.u = -self.K @ self.x_hat  # LQR control law
        
        # Increment time step
        self.t += 1

        # Add to history
        if self.t != 0:
            self.cov_hist.append(estimate_error)
            self.cost_hist.append(control_cost)
        
        return estimate_error, control_cost
    
    def plot_history(self):
        # Plot the history data
        fig,ax = plt.subplots(2)
        ax[0].plot(self.cov_hist)
        ax[0].set_title('Estimation error')
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Estimation error')
        ax[1].plot(self.cost_hist)
        ax[1].set_title('Control cost')
        ax[1].set_xlabel('Time')    
        ax[1].set_ylabel('Control cost')
        plt.tight_layout()
        plt.show()
    
    def simulate(self, T=50, plot=True):
        for t in range(T):
            epsilon, jota = self.step()
            print(f"Step: {self.t}, Estimation error: {epsilon}, Control cost: {jota}")
        
        if plot:
            self.plot_history() 
        
        return


# test with random values
if __name__ == "__main__":
    num_sensors = 4
    np.random.seed(42)  # For reproducibility

    A_S = np.array([
        [0.187, 0.475, 0.366, 0.299],
        [0.078, 0.078, 0.029, 0.433],
        [0.301, 0.354, 0.010, 0.485],
        [0.416, 0.106, 0.091, 0.092]
    ])

    B_Si = np.random.rand(num_sensors, num_sensors) * 0.1  # Small control influence
    C = np.random.rand(num_sensors, num_sensors)
    Q = np.eye(num_sensors) * 10  # Penalizing state deviation
    R = np.eye(num_sensors) * 10 # Control effort penalty
    W = np.eye(num_sensors) * 0.1  # Process noise
    V = np.eye(num_sensors) * 0.05  # Measurement noise
    M = np.random.rand(num_sensors, num_sensors)
    M = (M + M.T) / 2  # Make it symmetric

    x_initial = np.array([[-0.48], [-0.19], [-1.11], [-1.20]])  # Initial state

    # Define the LQG system and simulate
    lqg_system = LQG_EKF(None, A_S, B_Si, C, Q, R, W, V, M, num_sensors)
    lqg_system.x = x_initial  # Set initial state
    lqg_system.simulate(T=100)  # Run simulation for 10 time steps
