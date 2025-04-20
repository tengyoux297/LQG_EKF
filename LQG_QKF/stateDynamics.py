
import numpy as np
import scipy.linalg as la

def Vec(X):
    """Vectorize a matrix X (column-major ordering)."""
    return X.reshape(-1, 1, order='F') 

class StateDynamics(object):
  
  def __init__(self, n, p, W, A, B):
    """
    Initializes the class
    """
    self.x = np.zeros((n,1)) # state vector
    self.u = np.zeros((p,1)) # control input vector
    
    assert W.shape[0] == n, "W must be a square matrix of size n x n"
    assert W.shape[1] == n, "W must be a square matrix of size n x n"
    self.W = W # covariance matrix for process noise w
    self.w = np.zeros((n,1))
    self.n = n # state size
    self.p = p # control input size
    self.A = A.astype(np.float64) # state transition matrix
    self.B = B.astype(np.float64) # control input matrix
    self.t = 0 # time step
    self.trajectory = [] # trajectory of the system
    self.trajectory.append([self.x, self.u, self.w]) # append initial state and control input to trajectory
  def get_state_size(self):
    return self.n
 
  def get_input_size(self):
    return self.p
 
  def get_W(self):
    '''
    covariance matrix for process noise w
    '''
    return self.W
  
  def get_w(self):
    '''
    process noise
    '''
    return self.w
  
  def get_A(self):
    '''
    state transition matrix
    '''
    return self.A
  
  def get_B(self):
    '''
    control input matrix
    '''
    return self.B
  
  def get_current_state(self):
    '''
    current state vector
    '''
    return self.x 
  
  def get_current_control(self):
    '''
    current control input vector
    '''
    return self.u
  
  def get_traj_history(self):
    '''
    trajectory of the system
    '''
    return self.trajectory
  
  def forward(self, u):
    '''
    Forward kinematics of the system.
    '''
    self.u = u # update control input vector
    self.w = np.random.multivariate_normal(np.zeros(self.W.shape[0]), self.W)
    x1 = self.A @ self.x + self.B @ self.u + self.w # shape (n,1) 
    self.x = x1 # update state vector
    self.t += 1
    self.trajectory.append([self.x, self.u, self.w]) # append current state and control input to trajectory
    return self.t
  
# augmented system
  def aug_state(self):
    '''
    Augmented state vector
    '''
    x = self.x # shape (n,1)
    z1 = x # shape (n,1)
    z2 = Vec(x @ x.T) # shape (n^2, 1)
    z = (np.concatenate([z1.T, z2.T], axis=1)).T # shape (n+n^2, 1)
    return z, z1, z2
  
  def get_A_tilde(self):
    '''
    Augmented state transition matrix
    '''
    n = self.n # state size
    A = self.A # state transition matrix, shape (n,n)
    Bu = self.B @ self.u # shape (n,1)
    A11 = A # shape (n,n)
    A12 = np.zeros((n, n**2)) # shape (n,n^2)
    A21 = np.kron(Bu, A) + np.kron(A, Bu) # shape (n^2,n)
    A22 = np.kron(A, A) # shape (n^2,n^2)
    A_tilde = np.zeros(n+n**2, n+n**2) # shape (n+n^2,n+n^2)
    A_tilde[:n, :n] = A11
    A_tilde[:n, n:] = A12
    A_tilde[n:, :n] = A21
    A_tilde[n:, n:] = A22
    return A_tilde.astype(np.float64) # shape (n+n^2,n+n^2)

def get_B_tilde(self):
    '''
    Augmented control input matrix
    '''
    n = self.n # state size
    p = self.p # control input size
    B = self.B # control input matrix, shape (n,p)
    Bu = self.B @ self.u # shape (n,1)
    B_tilde = np.zeros((n+n**2, p)) # shape (n+n^2,p)
    B_tilde[:n, :] = B # shape (n,p)
    B_tilde[n:, :] = np.kron(Bu, np.eye(n)) # shape (n^2,p)
    return B_tilde.astype(np.float64) # shape (n+n^2,p)

class sensor(object):
  def __init__(self, C, M, V):
    """
    Initializes the class
    """
    # M shape - (m, n, n) where n is state size
    assert C.shape[0] == M.shape[0], "C and M must have the same length"
    self.m = M.shape[0] # number of measurements
    self.n = C.shape[1]
    
    self.M = M.astype(np.float64) # covariance matrix for process noise w
    self.V = V.astype(np.float64) # covariance matrix for measurement noise v
    self.C = C.astype(np.float64) # measurement matrix, shape (m, n)
    self.v = np.zeros((self.m,1)) # measurement noise vector
    
  def get_V(self):
    '''
    covariance matrix for process noise v
    '''
    return self.V
  
  def measure(self, x):
    '''
    measure the state vector x
    '''
    self.v = np.random.multivariate_normal(np.zeros(self.V.shape[0]), self.V)
    
    term1 = self.C @ x
    
    # x has shape (n, 1);  squeeze → (n,)
    x_vec = x.ravel()                       # (n,)

    # self.M has shape (m, n, n)
    # Compute q_i = xᵀ M[i] x  for all i = 0…m‑1 in one call:
    term2 = np.einsum('i, aij, j -> a',    # a ≡ output index (0…m‑1)
                      x_vec,               # i  ≡ first state index
                      self.M,              # a,i,j
                      x_vec)[:, None]      # j  ≡ second state index
                                          # result: (m,) → reshape to (m,1)
    return term1 + term2 + self.v # shape (m,1)
        
