"""
Application 4: Pendulum State Estimation with LQG-QKF and LQG-EKF
Option 3: Extended state vector with linearized dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import cholesky, sqrtm
from typing import Tuple, List, Dict, Any
import warnings
import sys
import os

# Add the correct paths to import LQG_QKF and stateDynamics
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, '2025_summer'))

from LQG_QKF import LQG
from stateDynamics import StateDynamics, sensor

warnings.filterwarnings('ignore')

# Global parameters
REPEAT = 1000  # Reduced for faster testing
NOISE1_REF = 1
SCALE = np.arange(-2, 2, 0.5)  # The range of measurement noise is 10^scale

def safe_matrix_inverse(A, reg=1e-8):
    """Safely compute matrix inverse with regularization"""
    try:
        return np.linalg.inv(A + reg * np.eye(A.shape[0]))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A + reg * np.eye(A.shape[0]))

class PendulumStateDynamics(StateDynamics):
    """
    Extended pendulum state dynamics with sin(θ) as additional state
    State vector: [ω, θ, sin(θ)] where ω is angular velocity, θ is angular position
    """
    
    def __init__(self, delta_t=0.01, len_pendulum=1.0, mass=1.0, g=9.8):
        self.delta_t = delta_t
        self.len_pendulum = len_pendulum
        self.mass = mass
        self.g = g
        
        # Extended state: [ω, θ, sin(θ)]
        n1 = 3  # Extended state size
        n2 = 0  # No sensor state
        p = 1   # Control input size
        
        # Process noise covariance
        sig_F = 1e-3
        W = np.array([[(sig_F*delta_t/mass/len_pendulum**2)**2, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]])
        
        # Linearized state transition matrix (will be updated)
        A_E = np.eye(3)
        A_S = np.zeros((0, 0))
        B_S = np.zeros((0, 1))
        
        super().__init__(n1, n2, p, W, A_E, A_S, B_S)
    
    def update_linearization(self, x):
        """
        Update the linearized dynamics around current state x
        """
        omega, theta, sin_theta = x.flatten()
        cos_theta = np.cos(theta)
        
        # Linearized state transition matrix
        self.A[:3, :3] = np.array([
            [1, -self.g/self.len_pendulum * cos_theta * self.delta_t, 0],
            [self.delta_t, 1, 0],
            [0, cos_theta * self.delta_t, 1]
        ])
        
        # Control input matrix (no direct control in this case)
        self.B[:3, 0] = np.array([0, 0, 0])
    
    def forward(self):
        """
        Forward step with non-linear dynamics
        """
        # Current state
        omega, theta, sin_theta = self.x_E.flatten()
        
        # Non-linear dynamics
        omega_dot = -self.g/self.len_pendulum * np.sin(theta)
        theta_dot = omega
        sin_theta_dot = np.cos(theta) * omega
        
        # Euler integration
        omega_new = omega + omega_dot * self.delta_t
        theta_new = theta + theta_dot * self.delta_t
        sin_theta_new = np.sin(theta_new)  # Keep sin(θ) consistent
        
        # Update state
        self.x_E = np.array([[omega_new], [theta_new], [sin_theta_new]])
        self.x = self.x_E  # No sensor state
        
        # Add process noise
        w = np.random.multivariate_normal([0, 0, 0], self.W)
        self.x_E += w.reshape(3, 1)
        
        # Update linearization
        self.update_linearization(self.x_E)
        
        # Update trajectory
        self.t += 1
        self.trajectory.append([self.x.copy(), self.u.copy()])
    
    def get_A_tilde(self):
        """Get augmented state transition matrix for QKF"""
        # For QKF, we need the augmented state transition matrix
        # This is a simplified version - in practice, this would be more complex
        A_tilde = np.eye(12)  # 3 + 9 = 12 (state + state squared terms)
        return A_tilde
    
    def get_Sigma_tilde(self):
        """Get augmented process noise covariance for QKF"""
        # Simplified augmented process noise
        Sigma_tilde = np.eye(12) * 0.01
        return Sigma_tilde
    
    def get_mu_tilde(self):
        """Get augmented state drift for QKF"""
        # Simplified augmented state drift
        mu_tilde = np.zeros((12, 1))
        return mu_tilde
    
    def get_z(self):
        """Get augmented state for QKF"""
        # Create augmented state [x, vec(xx')]
        x = self.x_E.flatten()
        xx = np.outer(x, x)
        z = np.concatenate([x, xx.flatten()])
        return z.reshape(-1, 1), None, None

class PendulumSensor(sensor):
    """
    Pendulum sensor with extended state measurement
    """
    
    def __init__(self, mass=1.0, len_pendulum=1.0, g=9.8, V=None):
        self.mass = mass
        self.len_pendulum = len_pendulum
        self.gravity = g  # Renamed from g to gravity to avoid conflict
        
        # Measurement matrix C for linear term
        # y = mg*cos(θ)*sin(θ) + ml*ω²*sin(θ)
        # With extended state [ω, θ, sin(θ)], this becomes:
        # y = mg*sin(θ)*sin(θ) + ml*ω²*sin(θ) = mg*sin²(θ) + ml*ω²*sin(θ)
        C = np.array([[0, 0, self.mass * self.gravity]])  # Linear term in sin(θ)
        
        # Quadratic term matrix M
        # The quadratic term ml*ω²*sin(θ) can be written as:
        # [ω, θ, sin(θ)]ᵀ * M * [ω, θ, sin(θ)]
        # where M captures the ω²*sin(θ) term
        M = np.array([[[self.mass * self.len_pendulum, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]]])  # Shape (1, 3, 3)
        
        # Measurement noise covariance
        if V is None:
            V = np.array([[1.0]])  # Will be scaled by noise level
        
        super().__init__(C, M, V)
    
    def g(self, x):
        """
        Measurement Jacobian (required by LQG framework)
        Returns the Jacobian of the measurement function with respect to state
        """
        omega, theta, sin_theta = x.flatten()
        cos_theta = np.cos(theta)
        
        # Jacobian of y = mg*sin²(θ) + ml*ω²*sin(θ) with respect to [ω, θ, sin(θ)]
        # ∂y/∂ω = 2*ml*ω*sin(θ)
        # ∂y/∂θ = 2*mg*sin(θ)*cos(θ) + ml*ω²*cos(θ)
        # ∂y/∂sin(θ) = 2*mg*sin(θ) + ml*ω²
        
        dydomega = 2 * self.mass * self.len_pendulum * omega * sin_theta
        dydtheta = 2 * self.mass * self.gravity * sin_theta * cos_theta + self.mass * self.len_pendulum * omega**2 * cos_theta
        dydsintheta = 2 * self.mass * self.gravity * sin_theta + self.mass * self.len_pendulum * omega**2
        
        return np.array([[dydomega, dydtheta, dydsintheta]])
    
    def measure(self, x):
        """
        Measurement function for extended state
        """
        omega, theta, sin_theta = x.flatten()
        
        # Original measurement: y = mg*cos(θ)*sin(θ) + ml*ω²*sin(θ)
        # With extended state: y = mg*sin²(θ) + ml*ω²*sin(θ)
        y = self.mass * self.gravity * sin_theta**2 + self.mass * self.len_pendulum * omega**2 * sin_theta
        
        # Add noise
        D = np.linalg.cholesky(self.V)
        noise = D @ np.random.standard_normal((1, 1))
        
        return np.array([[y]]) + noise
    
    def measure_pred(self, x_pred):
        """
        Measurement prediction for extended state
        """
        omega, theta, sin_theta = x_pred.flatten()
        
        # Same as measure but without noise
        y = self.mass * self.gravity * sin_theta**2 + self.mass * self.len_pendulum * omega**2 * sin_theta
        
        return np.array([[y]])
    
    def get_measA(self):
        """Get measurement offset for QKF"""
        return np.array([[0.0]])
    
    def get_aug_measB(self):
        """Get augmented measurement matrix for QKF"""
        # Simplified augmented measurement matrix
        # Shape: (1, 12) for 1 measurement and 12 augmented state dimensions
        B_tilde = np.zeros((1, 12))
        B_tilde[0, 2] = self.mass * self.gravity  # Linear term in sin(θ)
        B_tilde[0, 0] = self.mass * self.len_pendulum  # Quadratic term coefficient
        return B_tilde
    
    def aug_measure(self, z):
        """Augmented measurement function for QKF"""
        # Extract state from augmented state
        x = z[:3].flatten()
        omega, theta, sin_theta = x
        
        # Original measurement
        y = self.mass * self.gravity * sin_theta**2 + self.mass * self.len_pendulum * omega**2 * sin_theta
        
        # Add noise
        D = np.linalg.cholesky(self.V)
        noise = D @ np.random.standard_normal((1, 1))
        
        return np.array([[y]]) + noise

def simulation(noise1: float, repeat: int, filter_type: str) -> Tuple[float, float, float, List[float]]:
    """
    Main simulation function for pendulum state estimation with LQG framework
    
    Args:
        noise1: measurement noise magnitude
        repeat: number of Monte Carlo runs
        filter_type: 'qkf' or 'ekf'
    
    Returns:
        sitadot_rmse: RMSE for angular velocity
        sita_rmse: RMSE for angular position
        Runtime: average runtime per Monte Carlo run
        cost_history: list of cost-to-go values
    """
    # Problem setup
    delta_t = 0.01
    len_pendulum = 1
    mass = 1
    g = 9.8
    Tlimit = 1
    stepnum = int(Tlimit / delta_t)
    
    sitadot_rmse = 0
    sita_rmse = 0
    np.random.seed(123)
    Runtime = 0
    cost_history = []
    
    for iii in range(repeat):
        # Initialize true system
        omega_true = 0.0
        theta_true = np.pi/4
        sin_theta_true = np.sin(theta_true)
        x_true = np.array([[omega_true], [theta_true], [sin_theta_true]])
        
        # Initialize estimated system
        sig_init = np.array([np.pi/18, np.pi/18, 0.1])
        x_est = x_true + np.random.normal(0, sig_init).reshape(3, 1)
        
        # Create dynamics and sensor
        dynamics = PendulumStateDynamics(delta_t, len_pendulum, mass, g)
        sensor_obj = PendulumSensor(mass, len_pendulum, g, V=np.array([[noise1**2]]))
        
        # Initialize LQG system
        n1, n2, p = 3, 0, 1  # Extended state, no sensor state, 1 control input
        W = dynamics.W
        A_E = dynamics.A
        A_S = np.zeros((0, 0))
        B_S = np.zeros((0, 1))
        C = sensor_obj.C
        M = sensor_obj.M
        V = sensor_obj.V
        
        # Cost matrices for LQR control
        Q = np.eye(3) * 1.0  # State cost - penalize deviation from goal
        R = np.eye(1) * 0.1  # Control cost - penalize control effort
        
        # Goal state (pendulum at rest)
        goal_state = np.array([[0], [0], [0]])  # [ω=0, θ=0, sin(θ)=0]
        
        # Create LQG system with LQR control
        lqg = LQG(n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, goal_state, 
                  filter_type=filter_type, lqr_type='orig')
        
        # Set initial states
        dynamics.x_E = x_true.copy()
        dynamics.x = x_true.copy()
        lqg.x_hat = x_est.copy()
        lqg.F = dynamics
        lqg.sensor = sensor_obj
        
        # Initialize covariance
        if filter_type == 'qkf':
            # For QKF, need augmented state covariance
            P_init = np.eye(3) * 0.1
            lqg.Pz_est = np.block([[P_init, np.zeros((3, 9))],
                                   [np.zeros((9, 3)), np.eye(9) * 0.01]])
        else:
            # For EKF, standard covariance
            lqg.P_est = np.eye(3) * 0.1
        
        # Generate true trajectory and measurements
        true_trajectory = []
        measurements = []
        
        for i in range(stepnum):
            # True system step
            dynamics.forward()
            true_trajectory.append(dynamics.x.copy())
            
            # Generate measurement
            y = sensor_obj.measure(dynamics.x)
            measurements.append(y)
        
        # Estimation and control
        start_time = time.time()
        cost_trajectory = []
        
        for i in range(stepnum):
            # Update linearization around current estimate
            dynamics.update_linearization(lqg.x_hat)
            lqg.F = dynamics
            
            # LQE update (estimation)
            lqg.update_lqe()
            
            # LQR update (control)
            lqg.update_lqr()
            
            # Calculate cost-to-go based on estimated state
            state_error = lqg.x_hat - goal_state
            cost = state_error.T @ Q @ state_error
            cost_trajectory.append(cost[0, 0])
            
            # Forward step for next iteration
            dynamics.forward()
        
        temp = time.time() - start_time
        Runtime = Runtime + temp/repeat
        
        # Calculate RMSE for angular velocity and position
        omega_est = lqg.x_hat[0, 0]
        theta_est = lqg.x_hat[1, 0]
        
        omega_true_final = true_trajectory[-1][0, 0]
        theta_true_final = true_trajectory[-1][1, 0]
        
        sitadot_rmse = sitadot_rmse + (omega_true_final - omega_est)**2/repeat
        sita_rmse = sita_rmse + (theta_true_final - theta_est)**2/repeat
        
        # Store cost history for this run
        if iii == 0:  # Only store for first run to avoid averaging
            cost_history = cost_trajectory
    
    sita_rmse = np.sqrt(sita_rmse)
    sitadot_rmse = np.sqrt(sitadot_rmse)
    
    return sitadot_rmse, sita_rmse, Runtime, cost_history

def main():
    """
    Main function that runs simulations with LQG-QKF and LQG-EKF
    """
    print("Starting Pendulum State Estimation with LQG-QKF and LQG-EKF")
    print("This simulation may take several minutes...")
    
    # Initialize arrays for plotting
    xplotEKF = []
    yplotEKF = []
    Time_EKF = []
    xplotQKF = []
    yplotQKF = []
    Time_QKF = []
    cost_history_ekf = []
    cost_history_qkf = []
    
    for i in SCALE:
        noise1 = NOISE1_REF * 10**i
        print(f"Processing noise level {i+1}/{len(SCALE)}: magnitude = {noise1:.2e}")
        
        # EKF
        x_rmse, y_rmse, Runtime, cost_history = simulation(noise1, REPEAT, 'ekf')
        xplotEKF.append(x_rmse)
        yplotEKF.append(y_rmse)
        Time_EKF.append(Runtime)
        if i == 0:  # Store cost history for middle noise level
            cost_history_ekf = cost_history
        
        # QKF
        x_rmse, y_rmse, Runtime, cost_history = simulation(noise1, REPEAT, 'qkf')
        xplotQKF.append(x_rmse)
        yplotQKF.append(y_rmse)
        Time_QKF.append(Runtime)
        if i == 0:  # Store cost history for middle noise level
            cost_history_qkf = cost_history
    
    # Create plots
    plot_results({
        'xplotEKF': xplotEKF, 'yplotEKF': yplotEKF, 'Time_EKF': Time_EKF,
        'xplotQKF': xplotQKF, 'yplotQKF': yplotQKF, 'Time_QKF': Time_QKF,
        'cost_history_ekf': cost_history_ekf, 'cost_history_qkf': cost_history_qkf
    })
    
    return {
        'xplotEKF': xplotEKF, 'yplotEKF': yplotEKF, 'Time_EKF': Time_EKF,
        'xplotQKF': xplotQKF, 'yplotQKF': yplotQKF, 'Time_QKF': Time_QKF,
        'cost_history_ekf': cost_history_ekf, 'cost_history_qkf': cost_history_qkf
    }

def plot_results(results):
    """
    Create plots for the simulation results
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Angular Velocity RMSE
    ax1.semilogy(SCALE, results['xplotEKF'], 'b-o', label='EKF', linewidth=2, markersize=6)
    ax1.semilogy(SCALE, results['xplotQKF'], 'r-s', label='QKF', linewidth=2, markersize=6)
    ax1.set_xlabel('Measurement noise standard deviation (log scale)', fontsize=12)
    ax1.set_ylabel('Angular Velocity RMSE', fontsize=12)
    ax1.set_title('Pendulum State Estimation - Angular Velocity RMSE', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Angular Position RMSE
    ax2.semilogy(SCALE, results['yplotEKF'], 'b-o', label='EKF', linewidth=2, markersize=6)
    ax2.semilogy(SCALE, results['yplotQKF'], 'r-s', label='QKF', linewidth=2, markersize=6)
    ax2.set_xlabel('Measurement noise standard deviation (log scale)', fontsize=12)
    ax2.set_ylabel('Angular Position RMSE', fontsize=12)
    ax2.set_title('Pendulum State Estimation - Angular Position RMSE', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Runtime comparison
    ax3.plot(SCALE, results['Time_EKF'], 'b-o', label='EKF', linewidth=2, markersize=6)
    ax3.plot(SCALE, results['Time_QKF'], 'r-s', label='QKF', linewidth=2, markersize=6)
    ax3.set_xlabel('Measurement noise standard deviation', fontsize=12)
    ax3.set_ylabel('Average Runtime (seconds)', fontsize=12)
    ax3.set_title('Pendulum State Estimation - Runtime Comparison', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cost-to-Go History
    if len(results['cost_history_ekf']) > 0 and len(results['cost_history_qkf']) > 0:
        time_steps = range(len(results['cost_history_ekf']))
        ax4.plot(time_steps, results['cost_history_ekf'], 'b-', label='EKF', linewidth=2)
        ax4.plot(time_steps, results['cost_history_qkf'], 'r-', label='QKF', linewidth=2)
        ax4.set_xlabel('Time Step', fontsize=12)
        ax4.set_ylabel('Cost-to-Go', fontsize=12)
        ax4.set_title('Pendulum Control - Cost-to-Go History', fontsize=14)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')  # Log scale for better visualization
    else:
        ax4.text(0.5, 0.5, 'No cost data available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Cost-to-Go History (No Data)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('pendulum_state_estimation_lqg_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Simulation completed! Results saved to 'pendulum_state_estimation_lqg_results.png'")

if __name__ == "__main__":
    results = main() 