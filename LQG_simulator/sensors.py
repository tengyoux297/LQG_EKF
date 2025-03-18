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
        c = 1.0

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