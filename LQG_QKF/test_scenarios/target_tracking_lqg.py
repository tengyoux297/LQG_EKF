"""
Target Tracking Application - Python Conversion with LQG-QKF System
Adapted from: Y. Kim and H. Bang, Introduction to Kalman Filter and Its Applications, InTechOpen, 2018

Application 1: 3D target tracking with LQG-QKF system
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import cholesky, sqrtm
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Global parameters
REPEAT = 10  # Reduced from 10000 for faster testing
SCALE = np.arange(-4, 2.5, 0.5)  # Measurement noise range: 10^scale
I0 = -2  # Show convergence when std of measurement noise is 10^-2

def safe_matrix_inverse(A, reg=1e-8):
    """Safely compute matrix inverse with regularization"""
    try:
        return np.linalg.inv(A + reg * np.eye(A.shape[0]))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A + reg * np.eye(A.shape[0]))

def Vec(X):
    """Vectorize a matrix X (column-major ordering)."""
    return X.reshape(-1, 1, order='F') 

def invVec(X):
    """Inverse vectorization of a matrix X (column-major ordering)."""
    n = int(np.sqrt(X.shape[0]))
    return X.reshape(n, n, order='F')

def generate_random_symmetric_matrix(size, scale=1.0):
    """Generate a random symmetric positive definite matrix."""
    A = np.random.randn(size, size)
    return scale * (A.T @ A) + np.eye(size) * 1e-3

def finite_horizon_lqr(A, B, Q, R, N=100, Qf=None):
    """Finite horizon LQR solver."""
    if Qf is None:
        Qf = Q.copy()
    P = Qf.copy()
    # backward recursion
    for k in reversed(range(N)):
        P = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.pinv(R + B.T @ P @ B) @ B.T @ P @ A
    return P

class StateDynamics:
    """State dynamics for target tracking with LQG-QKF support."""
    
    def __init__(self, n1, n2, p, W, A_E, A_S, B_S):
        self.x_E = np.zeros((n1, 1))  # earth vector
        self.x_S = np.zeros((n2, 1))  # sensor vector
        
        n = n1 + n2  # state size
        self.n1 = n1  # earth state size
        self.n2 = n2  # sensor state size
        self.n = n  # state size
        self.x = np.vstack((self.x_E, self.x_S))  # state vector
        self.u = np.zeros((p, 1))  # control input vector
        
        self.W = W  # covariance matrix for process noise w
        self.p = p  # control input size
        
        self.A = np.zeros((n, n))  # state transition matrix
        self.A[:n1, :n1] = A_E
        self.A[n1:, n1:] = A_S
        
        self.B = np.zeros((n, p))  # control input matrix
        self.B[n1:, :p] = B_S  # shape (n2,p)
        self.t = 0  # time step
        self.trajectory = []  # trajectory of the system
        self.trajectory.append([self.x, self.u])
    
    def get_A(self):
        return self.A
    
    def get_B(self):
        return self.B
    
    def get_x(self):
        return self.x
    
    def set_u(self, u):
        self.u = u
    
    def get_u(self):
        return self.u
    
    def get_W(self):
        return self.W
    
    def get_w(self):
        """Process noise w (drawn fresh from W each time)."""
        omega = np.linalg.cholesky(self.W)
        rng_noise = np.random.default_rng()
        noise = omega @ rng_noise.standard_normal((self.n, 1))
        self.w = noise
        return noise
    
    def forward(self):
        """Forward kinematics of the system."""
        w = self.get_w()
        x1 = self.A @ self.x + self.B @ self.u + w
        self.x = x1
        self.x_E = x1[:self.n1]
        self.x_S = x1[self.n1:]
        self.t += 1
        self.trajectory.append([self.x, self.u])
        return self.t
    
    def get_z(self):
        """Get current augmented state vector."""
        x_vec = self.x
        xxT = x_vec @ x_vec.T
        z = np.vstack((x_vec, Vec(xxT)))
        return z, x_vec, xxT
    
    def get_mu_tilde(self):
        """Get mu_tilde for augmented system."""
        mu_tilde = np.zeros((self.n + self.n**2, 1))
        mu_tilde[:self.n] = self.B @ self.u
        return mu_tilde
    
    def get_Sigma_tilde(self):
        """Get Sigma_tilde for augmented system."""
        n = self.n
        Sigma_tilde = np.zeros((n + n**2, n + n**2))
        
        # Upper left block: W
        Sigma_tilde[:n, :n] = self.W
        
        # Upper right and lower left blocks: 0
        # Lower right block: W ⊗ W + W ⊗ (xx^T) + (xx^T) ⊗ W
        W_kron_W = np.kron(self.W, self.W)
        Sigma_tilde[n:, n:] = W_kron_W
        
        return Sigma_tilde
    
    def get_A_tilde(self):
        """Get A_tilde for augmented system."""
        n = self.n
        A_tilde = np.zeros((n + n**2, n + n**2))
        
        # Upper left block: A
        A_tilde[:n, :n] = self.A
        
        # Upper right block: 0
        # Lower left block: 0 (no direct coupling from x to xx^T)
        # Lower right block: A ⊗ A
        A_kron_A = np.kron(self.A, self.A)
        A_tilde[n:, n:] = A_kron_A
        
        return A_tilde

class Sensor:
    """Sensor with quadratic measurement functions."""
    
    def __init__(self, C, M, V):
        self.m = M.shape[0]  # number of measurements
        self.n = C.shape[1]
        
        self.M = M.astype(np.float64)  # quadratic measurement matrices
        self.V = V.astype(np.float64)  # covariance matrix for measurement noise
        self.C = C.astype(np.float64)  # linear measurement matrix
    
    def get_V(self):
        return self.V
    
    def get_measA(self):
        """Measurement matrix A term (zero in our case)."""
        return np.zeros((self.m, 1))
    
    def get_aug_measB(self):
        """Augmented measurement matrix B_tilde."""
        m, n = self.C.shape[0], self.C.shape[1]
        
        # Build the right-hand block: each row i is vec(M[i].T)
        right_term = np.zeros((m, n**2))
        for i in range(m):
            right_term[i] = Vec(self.M[i].T).squeeze()
        
        # Horizontal concatenation
        B_tilde = np.hstack((self.C, right_term))
        return B_tilde
    
    def measure(self, x):
        """Quadratic measurement function."""
        term1 = self.C @ x
        term2 = np.zeros((self.m, 1))
        for i in range(self.m):
            e = np.zeros((self.m, 1))
            e[i] = 1
            term2 += e @ x.T @ self.M[i] @ x
        
        D = np.linalg.cholesky(self.V)
        rng_noise = np.random.default_rng()
        term3 = D @ rng_noise.standard_normal((self.m, 1))
        return term1 + term2 + term3
    
    def measure_pred(self, x_pred):
        """Predicted measurement function."""
        term1 = self.C @ x_pred
        term2 = np.zeros((self.m, 1))
        for i in range(self.m):
            e = np.zeros((self.m, 1))
            e[i] = 1
            term2 += e @ x_pred.T @ self.M[i] @ x_pred
        return term1 + term2
    
    def g(self, x):
        """Measurement Jacobian."""
        term1 = self.C
        term2 = np.zeros((self.m, self.n))
        for i in range(self.m):
            e = np.zeros((self.m, 1))
            e[i] = 1
            term2 += e @ x.T @ self.M[i]
        return term1 + 2 * term2
    
    def aug_measure(self, z):
        """Augmented measurement function."""
        term1 = self.get_measA()
        term2 = self.get_aug_measB() @ z
        D = np.linalg.cholesky(self.V)
        rng_noise = np.random.default_rng()
        term3 = D @ rng_noise.standard_normal((self.m, 1))
        return term1 + term2 + term3

class LQGSystem:
    """LQG system with different filter types."""
    
    def __init__(self, n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, goal_state, H=50, 
                 filter_type='qkf', lqr_type='orig'):
        
        self.filter_type = filter_type
        self.lqr_type = lqr_type
        
        # Dynamics setting
        self.F = StateDynamics(n1, n2, p, W, A_E, A_S, B_S)
        n = n1 + n2
        
        # Sensor settings
        self.sensor = Sensor(C, M, V)
        self.V = self.sensor.get_V()
        
        # State settings
        self.A = self.F.get_A()
        self.B = self.F.get_B()
        self.n1 = n1
        self.n2 = n2
        self.n = n
        self.p = p
        self.W = self.F.get_W()
        
        # Augmented state settings (for QKF)
        mu_tilde_u = (np.eye(n+n**2) - self.F.get_A_tilde()).T @ self.F.get_mu_tilde()
        self.Z_est = mu_tilde_u
        I = np.eye(n**2 * (n+1)**2)
        Phi_tilde = self.F.get_A_tilde()
        Sigma_tilde = self.F.get_Sigma_tilde()
        vec_sigma_tilde_u = (I - np.kron(Phi_tilde, Phi_tilde)) @ Vec(Sigma_tilde)
        self.Pz_est = invVec(vec_sigma_tilde_u)
        
        # Horizon
        self.H = H
        
        # States
        self.x_hat = np.zeros((self.n, 1))
        self.z_hat = np.zeros((self.n + self.n**2, 1))
        self.x_goal = goal_state
        
        # LQR
        self.Q = Q.astype(np.float64)
        self.R = R.astype(np.float64)
        self.P_lqr = Q.copy()[:self.n, :self.n]
        
        # LQE
        self.P_est = np.eye(self.n) * 1e-6
    
    def update_lqr_analytic(self, goal_state, infinite_horizon=False):
        """Analytic LQR update for augmented state."""
        I_p = np.eye(self.p)  # shape (p, p)
        I_p2 = np.eye(self.p ** 2)  # shape (p^2, p^2)
        I_n = np.eye(self.n)  # shape (n, n)
        B = self.F.get_B()  # shape (n, p)
        A = self.F.get_A()  # shape (n, n)
        x = self.F.get_x()  # shape (n, 1), current state vector
        
        # commutation matrix for I_p kron u
        T = np.zeros((self.p * self.p, self.p * self.p))  # shape (p^2, p^2)
        for i in range(self.p):
            for j in range(self.p):
                e_ij = np.zeros((self.p, self.p))
                e_ij[i, j] = 1
                vec_e_ij = e_ij.T.flatten()  # transpose before vec
                T[:, i * self.p + j] = vec_e_ij

        M = np.kron(B, B) @ (I_p2 + T)  # shape (n^2, p^2)
        q = Vec(self.Q)  # shape (n^2, 1)
        
        S = np.zeros((self.p, self.p))  # shape (p, p)
        for i in range(self.p):
            e_i = np.zeros((self.p, 1))  # shape (p, 1)
            e_i[i] = 1
            term1 = (M @ np.kron(e_i, I_p))  # shape (n^2, p)
            term2 = term1 @ q @ e_i.T  # shape (p, p)
            S += term2  # accumulate over p columns

        Z = np.kron(A, B) @ np.kron(x, I_p) + np.kron(B, A) @ np.kron(I_p, x)  # shape (n^2, p)
        u_new = -np.linalg.inv(S + 2 * self.R) @ Z.T @ q  # shape (p, 1)
        self.F.set_u(u_new)
    
    def update_lqr_orig(self, goal_state):
        """Standard LQR update."""
        P_lqr = finite_horizon_lqr(self.A, self.B, self.Q[:self.n, :self.n], self.R, N=1, Qf=self.P_lqr)
        self.P_lqr = P_lqr.copy()
        feedback_gain = -np.linalg.pinv(self.R + self.B.T @ P_lqr @ self.B) @ self.B.T @ P_lqr @ self.A
        u_new = feedback_gain @ (goal_state - self.x_hat)
        self.F.set_u(u_new)
    
    def update_lqe_qkf(self):
        """QKF update."""
        Phi_tilde = self.F.get_A_tilde()
        Sigma_tilde = self.F.get_Sigma_tilde()
        mu_tilde = self.F.get_mu_tilde()
        
        # State prediction
        Z_pred = Phi_tilde @ self.Z_est + mu_tilde
        Pz_pred = Phi_tilde @ self.Pz_est @ Phi_tilde.T + Sigma_tilde
        
        # Measurement prediction
        measA = self.sensor.get_measA()
        measB_tilde = self.sensor.get_aug_measB()
        Y_pred = measA + measB_tilde @ Z_pred
        M = measB_tilde @ Pz_pred @ measB_tilde.T + self.V
        
        # Kalman gain
        K = Pz_pred @ measB_tilde.T @ np.linalg.inv(M)
        
        # State update
        Z, _, _ = self.F.get_z()
        Y_meas = self.sensor.aug_measure(Z)
        innovation = Y_meas - Y_pred
        self.Z_est = Z_pred + K @ innovation
        Pz_1 = Pz_pred - K @ M @ K.T
        
        self.Pz_est = Pz_1
        self.x_hat = self.Z_est[:self.n, :]
        return K
    
    def update_lqe_ekf(self):
        """EKF update."""
        mu = self.F.B @ self.F.u
        Phi = self.F.A
        Sigma = self.F.W
        
        # State prediction
        X_pred = mu + Phi @ self.x_hat
        P_pred = Phi @ self.P_est @ Phi.T + Sigma
        
        # Measurement prediction
        Y_pred = self.sensor.measure_pred(X_pred)
        g = self.sensor.g(X_pred)
        M = g @ P_pred @ g.T + self.sensor.V
        
        # Gain
        K = P_pred @ g.T @ np.linalg.inv(M)
        
        # State update
        Y_meas = self.sensor.measure(self.F.get_x())
        innov = Y_meas - Y_pred
        self.x_hat = X_pred + K @ innov
        self.P_est = P_pred - K @ M @ K.T
        return K
    
    def update_lqe_ukf(self):
        """UKF update."""
        # UKF parameters
        alpha = 1e-3
        beta = 2
        kappa = 0
        n = self.x_hat.shape[0]
        lambda_ = alpha**2 * (n + kappa) - n
        
        # Compute sigma points
        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = self.x_hat.flatten()
        
        try:
            sqrt_P = np.linalg.cholesky((n + lambda_) * self.P_est)
        except np.linalg.LinAlgError:
            eigenvals, eigenvecs = np.linalg.eigh(self.P_est)
            eigenvals = np.maximum(eigenvals, 1e-8)
            sqrt_P = eigenvecs @ np.diag(np.sqrt(eigenvals))
            sqrt_P = np.sqrt(n + lambda_) * sqrt_P
        
        for i in range(n):
            sigma_points[i + 1] = self.x_hat.flatten() + sqrt_P[i]
            sigma_points[n + i + 1] = self.x_hat.flatten() - sqrt_P[i]
        
        # Predict sigma points
        sigma_points_pred = np.zeros_like(sigma_points)
        for i in range(2 * n + 1):
            x_pred = self.F.A @ sigma_points[i].reshape(-1, 1) + self.F.B @ self.F.u
            sigma_points_pred[i] = x_pred.flatten()
        
        # Compute state mean
        weights_mean = np.full(2 * n + 1, 1 / (2 * (n + lambda_)))
        weights_mean[0] = lambda_ / (n + lambda_)
        x_predicted = np.sum(weights_mean[:, np.newaxis] * sigma_points_pred, axis=0).reshape(-1, 1)
        
        # Compute state covariance
        weights_cov = np.full(2 * n + 1, 1 / (2 * (n + lambda_)))
        weights_cov[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
        sigma_0 = self.F.W.copy()
        for i in range(2 * n + 1):
            diff = sigma_points_pred[i] - x_predicted.flatten()
            sigma_0 += weights_cov[i] * np.outer(diff, diff)
        
        # Predict measurements
        sigma_points_meas = np.zeros((2 * n + 1, self.sensor.m))
        for i in range(2 * n + 1):
            sigma_points_meas[i] = self.sensor.measure_pred(sigma_points_pred[i].reshape(-1, 1)).flatten()
        
        # Predict measurement mean
        y_predicted = np.sum(weights_mean[:, np.newaxis] * sigma_points_meas, axis=0).reshape(-1, 1)
        
        # Predict measurement covariance
        S = self.sensor.V.copy()
        for i in range(2 * n + 1):
            diff = sigma_points_meas[i] - y_predicted.flatten()
            S += weights_cov[i] * np.outer(diff, diff)
        
        # Cross covariance
        C_tilde = np.zeros((n, self.sensor.m))
        for i in range(2 * n + 1):
            diff_state = sigma_points_pred[i] - x_predicted.flatten()
            diff_meas = sigma_points_meas[i] - y_predicted.flatten()
            C_tilde += weights_cov[i] * np.outer(diff_state, diff_meas)
        
        # Kalman gain
        K = C_tilde @ np.linalg.pinv(S)
        
        # Measurement residual
        y = self.sensor.measure(self.F.get_x())
        delta_y = y - y_predicted
        
        # Update state estimate
        self.x_hat = x_predicted + K @ delta_y
        self.P_est = sigma_0 - K @ S @ K.T
        return K
    
    def update_lqe(self):
        """Update LQE based on filter type."""
        if self.filter_type == 'qkf':
            K = self.update_lqe_qkf()
        elif self.filter_type == 'ekf':
            K = self.update_lqe_ekf()
        elif self.filter_type == 'ukf':
            K = self.update_lqe_ukf()
        else:
            raise ValueError("Invalid filter type. Choose 'qkf', 'ekf', or 'ukf'.")
        return K
    
    def update_lqr(self):
        """Update LQR control based on filter and LQR type."""
        if self.lqr_type == 'orig':
            self.update_lqr_orig(self.x_goal)
        elif self.lqr_type == 'aug_analytic':
            self.update_lqr_analytic(self.x_goal)
    
    def forward_state(self):
        """Forward the state."""
        self.F.forward()
    
    def run_sim(self):
        """Run simulation with cost-to-go tracking."""
        rmse_list = []
        var_list = []
        cost_list = []
        
        for step in range(1, self.H + 1):
            self.update_lqe()
            self.update_lqr()
            self.forward_state()
            
            # Record error
            estimate_error = np.linalg.norm(self.F.get_x() - self.x_hat).item()
            rmse_list.append(estimate_error)
            
            # Record variance
            if self.filter_type == 'qkf':
                var = np.trace(self.Pz_est[:self.n, :self.n])
            else:
                var = np.trace(self.P_est)
            var_list.append(var)
            
            # Record cost
            x_goal = self.x_goal
            x_est = self.x_hat
            u = self.F.get_u()
            dx = x_est - x_goal
            cost = dx.T @ self.Q[:self.n, :self.n] @ dx + u.T @ self.R @ u
            cost_value = cost.item()
            cost_list.append(cost_value)
        
        # Calculate cost-to-go
        cost_to_go_list = []
        for i in range(len(cost_list)):
            cost_to_go = np.sum(cost_list[i:])
            cost_to_go_list.append(cost_to_go)
        
        return rmse_list, var_list, cost_to_go_list

def generate_stable_system_parameters(n1, n2, p, m, noise_scale=1e-1, m_scale=1e2):
    """Generate stable system parameters."""
    n = n1 + n2
    
    # Generate stable state transition matrices
    A_E = np.random.randn(n1, n1) * 0.1
    A_S = np.random.randn(n2, n2) * 0.1
    
    # Ensure stability
    eig_E, _ = np.linalg.eig(A_E)
    eig_S, _ = np.linalg.eig(A_S)
    
    max_eig_E = np.max(np.abs(eig_E))
    max_eig_S = np.max(np.abs(eig_S))
    
    if max_eig_E > 0.8:
        A_E = A_E * 0.8 / max_eig_E
    if max_eig_S > 0.8:
        A_S = A_S * 0.8 / max_eig_S
    
    B_S = np.random.randn(n2, p) * 0.1
    
    # Generate measurement matrices
    C = np.random.randn(m, n)
    
    # Generate quadratic measurement matrices
    M = []
    for i in range(m):
        M_i = generate_random_symmetric_matrix(n, scale=m_scale)
        M.append(M_i)
    M = np.array(M)
    
    # Generate noise matrices
    W = generate_random_symmetric_matrix(n, scale=noise_scale)
    V = generate_random_symmetric_matrix(m, scale=noise_scale)
    
    return A_E, A_S, B_S, C, M, W, V

def generate_goal_state(goal_state_E, state_S_size):
    """Generate goal state."""
    goal_state_S = np.random.randn(state_S_size, 1)
    goal_state = np.vstack((goal_state_E, goal_state_S))
    return goal_state

def simulation(magnitude: float, num_runs: int, KFtype: str, LQRtype: str = 'orig') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, List[float]]:
    """
    Main simulation function for target tracking with LQG-QKF system
    
    Args:
        magnitude: measurement noise magnitude
        num_runs: number of Monte Carlo runs
        KFtype: filter type ('ekf', 'ukf', 'qkf')
        LQRtype: LQR type ('orig', 'aug_analytic')
    
    Returns:
        time: time vector
        x_true: true state trajectory
        x_RMSE: RMSE for each state component
        p_sensor: sensor trajectory
        Runtime: average runtime per Monte Carlo run
        x_error_self_est: self-estimated error
        cost_to_go: cost-to-go curve for single run
    """
    np.random.seed(123)
    N = 30  # number of time steps
    dt = 1  # time between time steps
    
    # System parameters
    n1, n2, p, m = 2, 2, 3, 2  # state sizes and control/measurement dimensions
    
    sig_mea_true = np.array([1.0, 1.0]) * magnitude  # true measurement noise std
    sig_pro = np.array([1e-3, 1e-3, 1e-3])  # process noise std
    sig_mea = sig_mea_true  # user input measurement noise std
    sig_init = np.array([10, 10, 10, 0.1, 0.1, 0.1])  # initial guess std
    
    # Generate system parameters
    A_E, A_S, B_S, C, M, W, V = generate_stable_system_parameters(n1, n2, p, m, noise_scale=1e-1, m_scale=1e2)
    
    # Cost matrices
    Q = generate_random_symmetric_matrix(n1+n2, scale=1.0)
    R = generate_random_symmetric_matrix(p, scale=1.0)
    
    # Goal state
    goal_state = generate_goal_state(np.zeros((n1, 1)), n2)
    
    # Kalman filter simulation
    Runtime = 0
    res_x_est = np.zeros((n1+n2, N + 1, num_runs))  # Monte-Carlo estimates
    res_x_err = np.zeros((n1+n2, N + 1, num_runs))  # Monte-Carlo estimate errors
    P_diag = np.zeros((n1+n2, N + 1))  # diagonal term of error covariance matrix
    
    # Store cost-to-go for single run
    cost_to_go_single = []
    
    # Filtering
    for m_run in range(num_runs):
        # Create LQG system
        lqg_sys = LQGSystem(n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, goal_state, H=N+1, 
                           filter_type=KFtype, lqr_type=LQRtype)
        
        # Initial guess
        lqg_sys.x_hat = goal_state + np.random.normal(0, sig_init[:n1+n2])
        if KFtype == 'qkf':
            lqg_sys.Z_est[:n1+n2] = lqg_sys.x_hat
            lqg_sys.Pz_est[:n1+n2, :n1+n2] = np.diag(sig_init[:n1+n2]**2)
        else:
            lqg_sys.P_est = np.diag(sig_init[:n1+n2]**2)
        
        P_diag[:, 0] = np.diag(lqg_sys.P_est if KFtype != 'qkf' else lqg_sys.Pz_est[:n1+n2, :n1+n2])
        
        # Track cost-to-go for single run
        if m_run == 0:
            cost_to_go_single = []
        
        for k in range(1, N + 1):
            start_time = time.time()
            
            # Run one step of LQG
            lqg_sys.update_lqe()
            lqg_sys.update_lqr()
            lqg_sys.forward_state()
            
            temp_time = time.time() - start_time
            Runtime += temp_time / num_runs
            
            # Store results
            if KFtype == 'qkf':
                P_diag[:, k] = np.diag(lqg_sys.Pz_est[:n1+n2, :n1+n2])
                    else:
                P_diag[:, k] = np.diag(lqg_sys.P_est)
            
            # Calculate cost-to-go for single run
            if m_run == 0:
                x_goal = lqg_sys.x_goal
                x_est = lqg_sys.x_hat
                u = lqg_sys.F.get_u()
                dx = x_est - x_goal
                cost = dx.T @ lqg_sys.Q[:lqg_sys.n, :lqg_sys.n] @ dx + u.T @ lqg_sys.R @ u
                cost_to_go_single.append(cost.item())
        
        res_x_est[:, :, m_run] = lqg_sys.x_hat
        res_x_err[:, :, m_run] = lqg_sys.x_hat - lqg_sys.F.get_x()
    
    x_error_self_est = np.sqrt(P_diag[0, -1])
    time_vec = np.arange(0, N + 1) * dt
    
    # Calculate RMSE
    x_RMSE = np.zeros((n1+n2, N + 1))
    for k in range(N + 1):
        for i in range(n1+n2):
            x_RMSE[i, k] = np.sqrt(np.mean(res_x_err[i, k, :]**2))
    
    return time_vec, lqg_sys.F.get_x(), x_RMSE, np.zeros((3, N+1)), Runtime, x_error_self_est, cost_to_go_single

def main():
    """Main function to run the target tracking simulation with LQG-QKF system"""
    print("Starting Target Tracking Simulation with LQG-QKF System...")
    
    # Initialize result arrays
    results = {
        'EKF': {'x_rmse': [], 'vx_rmse': [], 'time': [], 'error_self_est': [], 'cost_to_go': []},
        'UKF': {'x_rmse': [], 'vx_rmse': [], 'time': [], 'error_self_est': [], 'cost_to_go': []},
        'QKF_orig': {'x_rmse': [], 'vx_rmse': [], 'time': [], 'error_self_est': [], 'cost_to_go': []},
        'QKF_analytic': {'x_rmse': [], 'vx_rmse': [], 'time': [], 'error_self_est': [], 'cost_to_go': []}
    }
    
    # Run simulations for each noise level
    for i, scale_val in enumerate(SCALE):
        magnitude = 10**scale_val
        print(f"Processing noise level {i+1}/{len(SCALE)}: magnitude = {magnitude:.2e}")
        
        # EKF
        time_vec, x_true, x_RMSE, p_sensor, Runtime, x_error_self_est, cost_to_go = simulation(magnitude, REPEAT, 'ekf', 'orig')
        results['EKF']['x_rmse'].append(x_RMSE[0, -1])
        results['EKF']['vx_rmse'].append(x_RMSE[3, -1])
        results['EKF']['time'].append(Runtime)
        results['EKF']['error_self_est'].append(x_error_self_est)
        if i == len(SCALE) // 2:  # Store cost-to-go for middle noise level
            results['EKF']['cost_to_go'] = cost_to_go
        
        # UKF
        time_vec, x_true, x_RMSE, p_sensor, Runtime, x_error_self_est, cost_to_go = simulation(magnitude, REPEAT, 'ukf', 'orig')
        results['UKF']['x_rmse'].append(x_RMSE[0, -1])
        results['UKF']['vx_rmse'].append(x_RMSE[3, -1])
        results['UKF']['time'].append(Runtime)
        results['UKF']['error_self_est'].append(x_error_self_est)
        if i == len(SCALE) // 2:  # Store cost-to-go for middle noise level
            results['UKF']['cost_to_go'] = cost_to_go
        
        # QKF with original LQR
        time_vec, x_true, x_RMSE, p_sensor, Runtime, x_error_self_est, cost_to_go = simulation(magnitude, REPEAT, 'qkf', 'orig')
        results['QKF_orig']['x_rmse'].append(x_RMSE[0, -1])
        results['QKF_orig']['vx_rmse'].append(x_RMSE[3, -1])
        results['QKF_orig']['time'].append(Runtime)
        results['QKF_orig']['error_self_est'].append(x_error_self_est)
        if i == len(SCALE) // 2:  # Store cost-to-go for middle noise level
            results['QKF_orig']['cost_to_go'] = cost_to_go
        
        # QKF with analytic LQR
        time_vec, x_true, x_RMSE, p_sensor, Runtime, x_error_self_est, cost_to_go = simulation(magnitude, REPEAT, 'qkf', 'aug_analytic')
        results['QKF_analytic']['x_rmse'].append(x_RMSE[0, -1])
        results['QKF_analytic']['vx_rmse'].append(x_RMSE[3, -1])
        results['QKF_analytic']['time'].append(Runtime)
        results['QKF_analytic']['error_self_est'].append(x_error_self_est)
        if i == len(SCALE) // 2:  # Store cost-to-go for middle noise level
            results['QKF_analytic']['cost_to_go'] = cost_to_go
    
    # Print runtime statistics
    print("\nRuntime Statistics (ms):")
    for key in results:
        if 'time' in results[key]:
            avg_time = np.mean(results[key]['time']) * 1000
            print(f"{key}: {avg_time:.2f} ms")
    
    # Create plots
    plot_results(results)
    
    return results

def plot_results(results):
    """Create performance comparison plots with cost-to-go curves"""
    noise_levels = 10**SCALE
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot 1: Position RMSE
    ax1 = axes[0]
    ax1.loglog(noise_levels, results['EKF']['x_rmse'], '-o', label='LQR+EKF', color='#0072BD', linewidth=1)
    ax1.loglog(noise_levels, results['UKF']['x_rmse'], '-s', label='LQR+UKF', color='#D95319', linewidth=1)
    ax1.loglog(noise_levels, results['QKF_orig']['x_rmse'], '-^', label='LQR+QKF (orig)', color='#EDB120', linewidth=1)
    ax1.loglog(noise_levels, results['QKF_analytic']['x_rmse'], '-d', label='LQR+QKF (analytic)', color='#7E2F8E', linewidth=1)
    
    ax1.set_ylabel('X-axis position RMSE (m)', fontsize=12)
    ax1.set_ylim([1e-3, 11])
    ax1.set_xlim([1e-4, 100])
    ax1.grid(True)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Velocity RMSE
    ax2 = axes[1]
    ax2.loglog(noise_levels, results['EKF']['vx_rmse'], '-o', label='LQR+EKF', color='#0072BD', linewidth=1)
    ax2.loglog(noise_levels, results['UKF']['vx_rmse'], '-s', label='LQR+UKF', color='#D95319', linewidth=1)
    ax2.loglog(noise_levels, results['QKF_orig']['vx_rmse'], '-^', label='LQR+QKF (orig)', color='#EDB120', linewidth=1)
    ax2.loglog(noise_levels, results['QKF_analytic']['vx_rmse'], '-d', label='LQR+QKF (analytic)', color='#7E2F8E', linewidth=1)
    
    ax2.set_ylabel('X-axis speed RMSE (m/s)', fontsize=12)
    ax2.set_ylim([1e-4, 1.5])
    ax2.set_xlim([1e-4, 100])
    ax2.grid(True)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 3: Cost-to-go comparison
    ax3 = axes[2]
    mid_idx = len(SCALE) // 2
    magnitude = 10**SCALE[mid_idx]
    
    # Use actual cost-to-go data from simulations
    time_steps = np.arange(1, len(results['EKF']['cost_to_go']) + 1)
    
    ax3.plot(time_steps, results['EKF']['cost_to_go'], '-o', label='LQR+EKF', color='#0072BD', linewidth=1)
    ax3.plot(time_steps, results['UKF']['cost_to_go'], '-s', label='LQR+UKF', color='#D95319', linewidth=1)
    ax3.plot(time_steps, results['QKF_orig']['cost_to_go'], '-^', label='LQR+QKF (orig)', color='#EDB120', linewidth=1)
    ax3.plot(time_steps, results['QKF_analytic']['cost_to_go'], '-d', label='LQR+QKF (analytic)', color='#7E2F8E', linewidth=1)
    
    ax3.set_xlabel('Time step', fontsize=12)
    ax3.set_ylabel('Cost-to-go', fontsize=12)
    ax3.set_title(f'Cost-to-go comparison (noise level: {magnitude:.2e})', fontsize=12)
    ax3.grid(True)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('target_tracking_lqg_qkf_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Results saved to 'target_tracking_lqg_qkf_results.png'")

if __name__ == "__main__":
    results = main() 