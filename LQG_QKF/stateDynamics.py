
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
  
  def set_control(self, u):
    '''
    set control input vector
    '''
    self.u = u
  
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
  
  def forward(self):
    '''
    Forward kinematics of the system.
    '''
    self.w = np.random.multivariate_normal(np.zeros(self.W.shape[0]), self.W).reshape(-1, 1) # process noise vector, shape (n,1)
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
  
  def mu_tilde(self): # mu_tilde
    Bu = self.B @ self.u # shape (n,1)
    term1 = Bu # shape (n,1)
    term2 = Vec(Bu @ Bu.T + self.W) # shape (n^2, 1) 
    u_tilde = np.vstack((term1, term2)) # shape (n+n^2, 1)
    return u_tilde # shape (n+n^2, 1)
  
  def aug_process_noise_covar(self):
    n = self.n # state size
    Sigma = self.W # shape (n,n)
    I_n = np.eye(n) # shape (n,n)
    Mu = self.B @ self.u # shape (n,1)
    # print(f'n: {n}, Mu: {Mu.shape}, Sigma: {Sigma.shape}')
    Gamma = np.kron(I_n, (Mu + self.A @ self.x)) + np.kron((Mu + self.A @ self.x), I_n) # shape (n^2,n)
    Lambda = np.zeros((n**2, n**2))
    for i in range(n):
        for j in range(n):
            row = i * n + j   # vec index of (i, j)
            col = j * n + i   # vec index of (j, i)
            Lambda[row, col] = 1   # single 1 per row/col
    Sigma11 = Sigma # shape (n,n)
    # print(f'Sigma: {Sigma.shape}, Gamma: {Gamma.shape}, Lambda: {Lambda.shape}')
    Sigma12 = Sigma @ Gamma.T # shape (n,n^2)
    Sigma21 = Gamma @ Sigma # shape (n^2,n)
    Sigma22 = Gamma @ Sigma @ Gamma.T + (np.eye(n**2) + Lambda) @ np.kron(Sigma, Sigma) # shape (n^2,n^2)
    
    Sigma_tilde = np.zeros((n+n**2, n+n**2)) # shape (n+n^2,n+n^2)
    Sigma_tilde[:n, :n] = Sigma11
    Sigma_tilde[:n, n:] = Sigma12
    Sigma_tilde[n:, :n] = Sigma21
    Sigma_tilde[n:, n:] = Sigma22
    return Sigma_tilde.astype(np.float64) # shape (n+n^2,n+n^2)
  
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
    A_tilde = np.zeros((n+n**2, n+n**2)) # shape (n+n^2,n+n^2)
    A_tilde[:n, :n] = A11
    A_tilde[:n, n:] = A12
    A_tilde[n:, :n] = A21
    A_tilde[n:, n:] = A22
    return A_tilde.astype(np.float64) # shape (n+n^2,n+n^2)

  def get_B_tilde(self): # B_tilde
    B = self.B # shape (n,p)
    u = self.u # shape (p,1)
    n, p = B.shape[0], B.shape[1] # state size, control input size
    # top block
    B_tilde_1 = B                        # (n, p)
    kron_sum = (np.kron(u, np.eye(p)) + np.kron(np.eye(p), u))        # shape (p^2, p)
    # B ⊗ B: shape (n^2, p^2)
    B_tilde_2 = np.kron(B,B) @ kron_sum                             # (n^2, p)
    B_tilde = np.vstack([B_tilde_1, B_tilde_2]) # (n+n^2, p)
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
    return self.V # shape (m,m)
  
  def get_measA(self):
    '''
    Augmented measurement matrix 1
    Actualy, this matrix does not exist in our case, but we need it for the QKF, so we define it as a zero matrix.
    '''
    return np.zeros((self.m, 1)) # shape (m,1)

  def get_aug_measB(self):
    '''
    Augmented measurement matrix 2
    '''
    if self.M.ndim != 3:
        raise ValueError("M_stack must be (m, n, n) or list of (n, n) matrices")

    m, n = self.C.shape[0], self.C.shape[1]  # m = number of measurements, n = state size
    if self.M.shape[1:] != (n, n) or self.M.shape[0] != m:
        raise ValueError("Inconsistent shapes between C and M_stack")

    # build the right-hand block: each row i is vec(M[i].T)
    M_vec_block = self.M.transpose(0, 2, 1).reshape(m, n * n)  # (m, n²)

    # horizontal concatenation
    C_tilde = np.hstack((self.C, M_vec_block)) # shape (m, n+n^2)
    return C_tilde    
  
  def measure(self, x):
    '''
    measure the state vector x
    '''
    self.v = np.random.multivariate_normal(np.zeros(self.V.shape[0]), self.V).reshape(-1, 1) # measurement noise vector, shape (m,1)
    
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
  
  def aug_measure(self, z):
    self.v = np.random.multivariate_normal(np.zeros(self.V.shape[0]), self.V)
    term1 = self.get_measA() # shape (m,1)
    term2 = self.get_aug_measB() @ z # shape (m,1)
    term3 = self.v # shape (m,1)
    
    return term1 + term2 + term3 # shape (m,1)