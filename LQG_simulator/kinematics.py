""" 
Implementation of the two-wheeled differential drive robot car 
and its controller.
 
Our goal in using LQR is to find the optimal control inputs:
  [linear velocity of the car, angular velocity of the car]
     
We want to both minimize the error between the current state 
and a desired state, while minimizing actuator effort 
(e.g. wheel rotation rate). These are competing objectives because a 
large u (i.e. wheel rotation rates) expends a lot of
actuator energy but can drive the state error to 0 really fast.
LQR helps us balance these competing objectives.
 
If a system is linear, LQR gives the optimal control inputs that 
takes a system's state to 0, where the state is
"current state - desired state".
 
Implemented by Addison Sears-Collins
Date: December 10, 2020
 
"""
# Import required libraries
import numpy as np
import scipy.linalg as la
 
class DifferentialDrive(object):
  """
  Implementation of Differential Drive kinematics.
  This represents a two-wheeled vehicle defined by the following states
  state = [x,y,theta] where theta is the yaw angle
  and accepts the following control inputs
  input = [linear velocity of the car, angular velocity of the car]
  """
  def __init__(self, noise_coeff=1.0):
    """
    Initializes the class
    """
    # Covariance matrix representing action noise 
    # (i.e. noise on the control inputs)
    self.V = None
    self.noise_coeff = noise_coeff
 
  def get_state_size(self):
    """
    The state is (X, Y) in global coordinates, so the state size is 2.
    """
    return 2
 
  def get_input_size(self):
    """
    The control input is ([X_velocity, Y_velocity]), 
    so the input size is 2.
    """
    return 2
 
  def get_V(self):
    """
    This function provides the covariance matrix V which 
    describes the noise that can be applied to the forward kinematics.
     
    Feel free to experiment with different levels of noise.
 
    Output
      :return: V: input cost matrix (2 x 2 matrix)
    """
    # The np.eye function returns a 2D array with ones on the diagonal
    # and zeros elsewhere.
    if self.V is None:
      self.V = np.eye(2)
      self.V[0,0] = 0.01
      self.V[1,1] = 0.01
    return self.noise_coeff * self.V
 
  def forward(self,x0,u,v,dt=0.1):
    """
    Computes the forward kinematics for the system.
 
    Input
      :param x0: The starting state (position) of the system (units:[m,m,rad])  
                 np.array with shape (3,) -> 
                 (X, Y, THETA)
      :param u:  The control input to the system  
                 2x1 NumPy Array given the control input vector is  
                 [linear velocity of the car, angular velocity of the car] 
                 [meters per second, radians per second]
      :param v:  The noise applied to the system (units:[m, m, rad]) ->np.array 
                 with shape (3,)
      :param dt: Change in time (units: [s])
 
    Output
      :return: x1: The new state of the system (X, Y, THETA)
    """
    u0 = u 
         
    # If there is no noise applied to the system
    if v is None:
      v = 0
  
    # Control input
    # Velocity in the x and y direction in m/s
    x_dot = u0[0]
    y_dot = u0[1]
   
    # The new state of the system
    x1 = np.empty(2)
     
    # Calculate the new state of the system
    # Noise is added like in slide 34 in Lecture 7
    x1[0] = x0[0] + x_dot * dt + v[0] # X
    x1[1] = x0[1] + y_dot * dt + v[1] # Y
 
    return x1
 
  def linearize(self, dt=0.1):
    """
    Creates a linearized version of the dynamics of the differential 
    drive robotic system (i.e. a
    robotic car where each wheel is controlled separately.
 
    The system's forward kinematics are nonlinear due to the sines and 
    cosines, so we need to linearize 
    it by taking the Jacobian of the forward kinematics equations with respect 
     to the control inputs.
 
    Our goal is to have a discrete time system of the following form: 
    x_t+1 = Ax_t + Bu_t where:
 
    Input
      :param x: The state of the system (units:[m,m,rad]) -> 
                np.array with shape (3,) ->
                (X, Y, THETA) ->
                X_system = [x1, x2, x3]
      :param dt: The change in time from time step t to time step t+1      
 
    Output
      :return: A: Matrix A is a 2x2 matrix (because there are 2 states) that 
                  describes how the state of the system changes from t to t+1 
                  when no control command is executed. Typically, 
                  a robotic car only drives when the wheels are turning. 
                  Therefore, in this case, A is the identity matrix.
      :return: B: Matrix B is a 2 x 2 matrix (because there are 2 states and 
                  2 control inputs) that describes how
                  the state (X, Y) changes from t to t + 1 due to 
                  the control command u.
                  Matrix B is found by taking the The Jacobian of the three 
                  forward kinematics equations (for X, Y) 
                  with respect to u (2 x 1 matrix)
 
    """
    ####### A Matrix #######
    # A matrix is the identity matrix
    A = np.array([[1.0,   0],
                  [  0, 1.0]])
 
    ####### B Matrix #######
    B = np.array([[dt, 0],
                  [0, dt]])
         
    return A, B

 
def get_R():
    """
    This function provides the R matrix to the lqr_ekf_control simulator.
     
    Returns the input cost matrix R.
 
    Experiment with different gains.
    This matrix penalizes actuator effort 
    (i.e. rotation of the motors on the wheels).
    The R matrix has the same number of rows as are actuator states 
    [linear velocity of the car, angular velocity of the car]
    [meters per second, radians per second]
    This matrix often has positive values along the diagonal.
    We can target actuator states where we want low actuator 
    effort by making the corresponding value of R large.   
 
    Output
      :return: R: Input cost matrix
    """
    R = np.array([[0.01, 0],  # Penalization for linear velocity effort
                  [0, 0.01]]) # Penalization for angular velocity effort
 
    return R
 
def get_Q():
    """
    This function provides the Q matrix to the lqr_ekf_control simulator.
     
    Returns the state cost matrix Q.
 
    Experiment with different gains to see their effect on the vehicle's 
    behavior.
    Q helps us weight the relative importance of each state in the state 
    vector (X, Y, THETA). 
    Q is a square matrix that has the same number of rows as there are states.
    Q penalizes bad performance.
    Q has positive values along the diagonal and zeros elsewhere.
    Q enables us to target states where we want low error by making the 
    corresponding value of Q large.
    We can start with the identity matrix and tweak the values through trial 
    and error.
 
    Output
      :return: Q: State cost matrix (3x3 matrix because the state vector is 
                  (X, Y, THETA))
    """
    Q = np.array([[0.4, 0],  # Penalize X position error (global coordinates)
                  [0, 0.4]]) # Penalize Y position error (global coordinates)
     
    return Q