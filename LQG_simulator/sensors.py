"""
Extended Kalman Filter implementation using landmarks to localize
 
Implemented by Addison Sears-Collins
Date: December 10, 2020
"""
import numpy as np
import scipy.linalg as la
import math
 
class LandmarkDetector(object):
  """
  This class represents the sensor mounted on the robot that is used to 
  to detect the location of landmarks in the environment.
  """
  def __init__(self,landmark_list):
    """
    Calculates the sensor measurements for the landmark sensor
         
    Input
      :param landmark_list: 2D list of landmarks [L1,L2,L3,...,LN], 
                            where each row is a 2D landmark defined by 
                                        Li =[l_i_x,l_i_y]
                            which corresponds to its x position and y 
                            position in the global coordinate frame.
    """
    # Store the x and y position of each landmark in the world
    self.landmarks = np.array(landmark_list) 
 
    # Record the total number of landmarks
    self.N_landmarks = self.landmarks.shape[0]
         
    # Variable that represents landmark sensor noise (i.e. error)
    self.W = None
 
  def get_W(self):
    """
    This function provides the covariance matrix W which describes the noise 
    that can be applied to the sensor measurements.
 
    Feel free to experiment with different levels of noise.
     
    Output
      :return: W: measurement noise matrix (4x4 matrix given two landmarks)
                  In this implementation there are two landmarks. 
                  Each landmark has an x location and y location, so there are 
                  4 individual sensor measurements. 
    """
    # In the EKF code, we will condense this 4x4 matrix so that it is a 4x1 
    # landmark sensor measurement noise vector (i.e. [x1 noise, y1 noise, x2 
    # noise, y2 noise]
    if self.W is None:
      self.W = 1e-1*np.eye(2*self.N_landmarks)
    return self.W
 
import numpy as np

class LandmarkDetector(object):
    """
    This class represents the sensor mounted on the robot that is used to 
    to detect the location of landmarks in the environment.
    """
    def __init__(self, landmark_list, noise_coeff = 1):
        """
        Calculates the sensor measurements for the landmark sensor
        
        Input
          :param landmark_list: 2D list of landmarks [L1, L2, L3, ..., LN], 
                                where each row is a 2D landmark defined by 
                                Li = [l_i_x, l_i_y]
                                which corresponds to its x position and y 
                                position in the global coordinate frame.
        """
        # Store the x and y position of each landmark in the world
        self.landmarks = np.array(landmark_list)
        
        # Record the total number of landmarks
        self.N_landmarks = self.landmarks.shape[0]
        
        # Variable that represents landmark sensor noise (i.e. error)
        self.W = None
        self.noise_coeff = noise_coeff
    
    def get_W(self):
        """
        This function provides the covariance matrix W which describes the noise 
        that can be applied to the sensor measurements.
        
        Feel free to experiment with different levels of noise.
        
        Output
          :return: W: measurement noise matrix (4x4 matrix given two landmarks)
                      In this implementation there are two landmarks. 
                      Each landmark has an x location and y location, so there are 
                      4 individual sensor measurements. 
        """
        # In the EKF code, we will condense this 4x4 matrix so that it is a 4x1 
        # landmark sensor measurement noise vector (i.e. [x1 noise, y1 noise, x2 
        # noise, y2 noise]
        if self.W is None:
            self.W = self.noise_coeff * np.eye(self.N_landmarks)
        return self.W
    
    def measure(self, x, w=None):
        """
        Computes the landmark sensor measurements for the system. 
        This will be a list of quadratic measurements between the robot and a 
        set of landmarks defined with [x, y] positions.

        Input
          :param x: The state [x_t, y_t, theta_t] of the system -> np.array with 
                    shape (3,). x is a 3x1 vector
          :param w: The noise (assumed Gaussian) applied to the measurement -> 
                    np.array with shape (N_Landmarks,)
                    w is a 2x1 vector
        Output
          :return: y: The resulting observation vector (2x1 in this 
                      implementation because there are 2 landmarks). The 
                      resulting measurement is 
                      y = [h1, h2]
        """
        # Create the observation vector
        y = np.zeros(shape=(self.N_landmarks,))

        # Initialize state variables    
        x_t = x[0]
        y_t = x[1]
        
        # Constants for the quadratic function
        a = 1.0
        b = 1.0
        c = 0.0

        # Fill in the observation model y
        for i in range(0, self.N_landmarks):

            # Set the x and y position for landmark i
            x_l_i = self.landmarks[i][0]
            y_l_i = self.landmarks[i][1]

            # Set the noise value
            w_i = w[i] if w is not None else 0.0

            # Calculate the quadratic measurement
            y_i = a * (x_t - x_l_i)**2 + b * (y_t - y_l_i)**2 + c + w_i

            # Populate the predicted sensor observation vector      
            y[i] = y_i

        return y
    
    def jacobian(self, x):
        """
        Computes the first order jacobian around the state x

        Input
          :param x: The starting state (position) of the system -> np.array 
                    with shape (3,)

        Output
          :return: H: The resulting Jacobian matrix H for the sensor. 
                      The number of rows is equal to the number of landmarks.
                      The number of columns is equal to the number of states (X, Y, THETA).
        """
        # Create the Jacobian matrix H
        H = np.zeros(shape=(self.N_landmarks, 3))

        # Extract the state
        X_t = x[0]
        Y_t = x[1]
        
        # Constants for the quadratic function
        a = 1.0
        b = 1.0

        # Fill in H, the Jacobian matrix with the partial derivatives of the 
        # observation (measurement) model h
        for i in range(0, self.N_landmarks):
        
            # Set the x and y position for landmark i
            x_l_i = self.landmarks[i][0]
            y_l_i = self.landmarks[i][1]
        
            H[i, 0] = 2 * a * (X_t - x_l_i)
            H[i, 1] = 2 * b * (Y_t - y_l_i)
            H[i, 2] = 0

        return H
 
def EKF(DiffDrive,Sensor,y,x_hat,sigma,u,dt,V=None,W=None):
    """
    Some of these matrices will be non-square or singular. Utilize the 
    pseudo-inverse function la.pinv instead of inv to avoid these errors.
 
    Input
      :param DiffDrive: The DifferentialDrive object defined in kinematics.py
      :param Sensor: The Landmark Sensor object defined in this class
      :param y: The observation vector (4x1 in this implementation because 
                there are 2 landmarks). The measurement is 
                y = [h1_range_1,h1_angle_1, h2_range_2, h2_angle_2]
      :param x_hat: The starting estimate of the state at time t -> 
                    np.array with shape (3,)
                    (X_t, Y_t, THETA_t)
      :param sigma: The state covariance matrix at time t -> np.array with 
                    shape (3,1) initially, then 3x3
      :param u: The input to the system at time t -> np.array with shape (2,1)
                These are the control inputs to the system.
                [left wheel rotational velocity, right wheel rotational 
                velocity]
      :param dt: timestep size delta t
      :param V: The state noise covariance matrix  -> np.array with shape (3,3)
      :param W: The measurment noise covariance matrix -> np.array with shape 
                (2*N_Landmarks,2*N_Landmarks)
                4x4 matrix
 
    Output
      :return: x_hat_2: The estimate of the state at time t+1 
                        [X_t+1, Y_t+1, THETA_t+1]
      :return: sigma_est: The new covariance matrix at time t+1 (3x3 matrix)
       
    """
    V = DiffDrive.get_V() # 3x3 matrix for the state noise
    W = Sensor.get_W() # 4x4 matrix for the measurement noise
 
    ## Generate noise
    # v = process noise, w = measurement noise
    v = np.random.multivariate_normal(np.zeros(V.shape[0]),V) # 3x1 vector
    w = np.random.multivariate_normal(np.zeros(W.shape[0]),W) # 4x1 vector  
     
    ##### Prediction Step #####
 
    # Predict the state estimate based on the previous state and the 
    # control input
    # x_predicted is a 3x1 vector (X_t+1, Y_t+1, THETA_t+1)
    x_predicted = DiffDrive.forward(x_hat,u,v,dt)
 
    # Calculate the A and B matrices    
    A, B = DiffDrive.linearize(x=x_hat)
     
    # Predict the covariance estimate based on the 
    # previous covariance and some noise
    # A and V are 3x3 matrices
    sigma_3by3 = None
    if (sigma.size == 3):
      sigma_3by3 = sigma * np.eye(3)
    else:
      sigma_3by3 = sigma
       
    sigma_new = A @ sigma_3by3 @ A.T + V
 
    ##### Correction Step #####  
 
    # Get H, the 4x3 Jacobian matrix for the sensor
    H = Sensor.jacobian(x_hat) 
 
    # Calculate the observation model   
    # y_predicted is a 4x1 vector
    y_predicted = Sensor.measure(x_predicted, w)
 
    # Measurement residual (delta_y is a 4x1 vector)
    delta_y = y - y_predicted
 
    # Residual covariance 
    # 4x3 @ 3x3 @ 3x4 -> 4x4 matrix
    S = H @ sigma_new @ H.T + W
 
    # Compute the Kalman gain
    # The Kalman gain indicates how certain you are about
    # the observations with respect to the motion
    # 3x3 @ 3x4 @ 4x4 -> 3x4
    K = sigma_new @ H.T @ la.pinv(S)
 
    # Update the state estimate
    # 3x1 + (3x4 @ 4x1 -> 3x1)
    x_hat_2 = x_predicted + (K @ delta_y)
     
    # Update the covariance estimate
    # 3x3 - (3x4 @ 4x3) @ 3x3
    sigma_est = sigma_new - (K @ H @ sigma_new)
 
    return x_hat_2 , sigma_est