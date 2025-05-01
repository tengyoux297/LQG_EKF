import numpy as np

def Vec(X):
        """Vectorize a matrix X (column-major ordering)."""
        return X.reshape(-1, 1, order='F') 
    
class NewtonSolver(object):
    def __init__(self, n, p):
        """
        Initializes the class with parameters.
        """
        self.n = n
        self.p = p

    def term2(self, u, params):
        """
        Implements:
        ∂/∂u(t) [ z^T A_tilde^T P^z A_tilde z ]
        = z1^T (A ⊗ B + B ⊗ A)^T Pz21 A z1
            + z1^T (A ⊗ B + B ⊗ A) Pz12^T A z1
            + z1^T (A ⊗ B + B ⊗ A)^T Pz22 (A ⊗ A) z2
            + z1^T (A ⊗ B + B ⊗ A)^T Pz22^T (A ⊗ A) z2
            + z1^T (A ⊗ B + B ⊗ A)^T Pz22 (A ⊗ (B u) + (B u) ⊗ A) z1
            + z1^T (A ⊗ B + B ⊗ A)^T Pz22^T (A ⊗ (B u) + (B u) ⊗ A) z1
        """
        n, p = self.n, self.p
        z1   = params['z1']    # z₁(t), assumed shape (n,)
        z2   = params['z2']    # z₂(t), assumed shape (n^2, 1)
        A    = params['A']     # matrix A, shape (n, n)
        B    = params['B']     # matrix B, shape (n, p)
        Pz21 = params['Pz21']  # block of P^z(t+1), shape (n^2, n)
        Pz12 = params['Pz12']  # block of P^z(t+1), shape (n, n^2)
        Pz22 = params['Pz22']  # block of P^z(t+1), shape (n^2, n^2)
        
        Bu = B @ u  # shape (n,1)
        
        # K = (A ⊗ B + B ⊗ A)
        K = np.kron(A, B) + np.kron(B, A) # shape (n^2, np)
        
        T2_1 = np.einsum('ij,jkq->ikq', z1.T, np.reshape(K.T, (n, p, n**2))) @ Pz21 @ (A @ z1)
        T2_2 = np.einsum('ijk,kpn->ijn', np.einsum('ij,jkpn->ipn', z1.T, np.reshape(K, (n, n, p, n))), np.reshape(Pz12.T, (n,n,n))) @ A @ z1
        
        kron_AA = np.kron(A, A) # shape (n^2, n^2)
        T2_3 = np.einsum('ij,jkq->ikq', z1.T, np.reshape(K.T, (n, p, n**2))) @ Pz22 @ kron_AA @ z2
        T2_4 = np.einsum('ij,jkq->ikq', z1.T, np.reshape(K.T, (n, p, n**2))) @ Pz22.T @ kron_AA @ z2
        
        # Compute (A ⊗ (B u) + (B u) ⊗ A)
        kron_term = np.kron(A, Bu) + np.kron(Bu, A) # shape (n^2, n)
        
        T2_5 = np.einsum('ij,jkq->ikq', z1.T, np.reshape(K.T, (n, p, n**2))) @ Pz22 @ kron_term @ z1
        T2_6 = np.einsum('ij,jkq->ikq', z1.T, np.reshape(K.T, (n, p, n**2))) @ Pz22.T @ kron_term @ z1
        
        # sum all components
        result = T2_1 + T2_2 + T2_3 + T2_4 + T2_5 + T2_6
        return result

    def term3(self, u, params):
        """
        Implements the derivative:
        ∂/∂u [ 2 z(t)^T A_tilde(t)^T P^z(t+1) u_tilde(t) ]
        = 2 [ B^T Pz11^T A z1 
            + B^T Pz21 ( (A⊗A) z2 )
            + B^T Pz21^T ( (A⊗(B u) + (B u)⊗A) z1 )
            + z1^T ( (A⊗B + B⊗A)^T Pz21 B u )
            + ( (I_p⊗u^T + u^T⊗I_p) ( (B⊗B)^T Pz12^T A z1 ) )
            + ( (I_p⊗u^T + u^T⊗I_p) ( (B⊗B)^T Pz22^T ( (A⊗(B u) + (B u)⊗A) z1 ) ) )
            + ( (I_p⊗u^T + u^T⊗I_p) ( (B⊗B)^T Pz22^T ( (A⊗A) z2 ) ) )
            + z1^T ( (A⊗B + B⊗A)^T Pz22 Vec(B u u^T B^T + W) ) ]
        """
        n, p = self.n, self.p
        z1 = params['z1']         # (n,)
        A  = params['A']          # (n x n)
        B  = params['B']          # (n x p)
        Pz11 = params['Pz11']
        Pz21 = params['Pz21']
        Pz12 = params['Pz12']
        Pz22 = params['Pz22']
        W = params['W']
        p = B.shape[1]
        
        Bu = B @ u               # (n,)
        z2 = params['z2']        # (n^2,) (assuming proper stacking)
        
        # T3_1: 2 * B^T Pz11^T A z1
        T3_1 = 2 * (B.T @ Pz11.T @ (A @ z1))
        # T3_2: 2 * B^T Pz21 ( (A⊗A) z2 )
        T3_2 = 2 * (np.einsum('ij,jkq->iq', np.einsum('ij,jkq->iq', B.T, np.reshape(Pz21, (n,n,n))), np.reshape(np.kron(A, A), (n,n,n**2))) @ z2)
        # T3_3: 2 * B^T Pz21^T ( (A⊗(B u) + (B u)⊗A) z1 )
        T3_3 = 2 * (B.T @ Pz21.T @ ((np.kron(A, Bu) + np.kron(Bu, A)) @ z1))
        # T3_4: 2 * z1^T ( (A⊗B + B⊗A)^T Pz21 (B u) )
        T3_4 = 2 * (np.einsum('ij,jkq->ikq', z1.T, np.reshape((np.kron(A, B) + np.kron(B, A)).T, (n,p,n**2))) @ Pz21 @ (B @ u))
        
        I_p = np.eye(p)
        kron_term = np.kron(B, B)
        # T3_5: 2 * ( (I_p⊗u^T + u^T⊗I_p) ( kron(B, B)^T Pz12^T A z1 ) )
        T3_5 = 2 * ((np.kron(I_p, u.T) + np.kron(u.T, I_p)) @ (kron_term.T @ Pz12.T @ (A @ z1)))
        # T3_6: 2 * ( (I_p⊗u^T + u^T⊗I_p) ( kron(B, B)^T Pz22^T ( (np.kron(A, Bu)+np.kron(Bu, A)) z1 ) ) )
        T3_6 = 2 * ((np.kron(I_p, u.T) + np.kron(u.T, I_p)) @ (kron_term.T @ Pz22.T @ ((np.kron(A, Bu) + np.kron(Bu, A)) @ z1)))
        # T3_7: 2 * ( (I_p⊗u^T + u^T⊗I_p) ( kron(B, B)^T Pz22^T ( (A⊗A) z2 ) ) )
        T3_7 = 2 * ((np.kron(I_p, u.T) + np.kron(u.T, I_p)) @ (kron_term.T @ Pz22.T @ (np.kron(A, A) @ z2)))
        # T3_8: 2 * z1^T ( (A⊗B + B⊗A)^T Pz22 Vec(B u u^T B^T + W) )
        # approximate B u u^T B^T by outer(Bu, Bu)
        T3_8 = 2 * (np.einsum('ij,jkq->ikq', z1.T, np.reshape((np.kron(A, B) + np.kron(B, A)).T, (n,p,n**2))) @ Pz22 @ Vec(B @ u @ u.T @ B.T + W))
        return T3_1 + T3_2 + T3_3 + T3_4 + T3_5 + T3_6 + T3_7 + T3_8

    def term4(self, u, params):
        """
        Implements the derivative:
        ∂/∂u [ ũ(t)^T P ũ(t) ]
        = 2 B^T Pz11 B u
        + (I_p⊗u^T + u^T⊗I_p)(B⊗B)^T Pz21 B u
        + (I_p⊗u^T + u^T⊗I_p)(B⊗B)^T Pz12^T B u
        + B^T Pz21^T Vec(B u u^T B^T + W)
        + B^T Pz12 Vec(B u u^T B^T + W)
        + (I_p⊗u^T + u^T⊗I_p)(B⊗B)^T Pz22 Vec(B u u^T B^T + W)
        + (I_p⊗u^T + u^T⊗I_p)(B⊗B)^T Pz22^T Vec(B u u^T B^T + W)
        """
        B = params['B']         # (n x p)
        p = B.shape[1]
        Pz11 = params['Pz11']
        Pz21 = params['Pz21']
        Pz12 = params['Pz12']
        Pz22 = params['Pz22']
        W = params['W']
        
        Bu = B @ u              # (n,)
        I_p = np.eye(p)
        kron_term = np.kron(B, B)
        
        # T4_1: 2 * B^T Pz11 B u
        T4_1 = 2 * (B.T @ Pz11 @ (B @ u))
        # T4_2: (I_p⊗u^T + u^T⊗I_p)(B⊗B)^T Pz21 B u
        T4_2 = (np.kron(I_p, u.T) + np.kron(u.T, I_p)) @ (kron_term.T @ Pz21 @ (B @ u))
        # T4_3: (I_p⊗u^T + u^T⊗I_p)(B⊗B)^T Pz12^T B u
        T4_3 = (np.kron(I_p, u.T) + np.kron(u.T, I_p)) @ (kron_term.T @ Pz12.T @ (B @ u))
        
        # Vec(B u u^T B^T + W)
        vec_term = Vec(np.outer(Bu, Bu) + W)
        
        # T4_4: B^T Pz21^T Vec(B u u^T B^T + W)
        T4_4 = B.T @ Pz21.T @ vec_term
        # T4_5: B^T Pz12 vec_term
        T4_5 = B.T @ Pz12 @ vec_term
        # T4_6: (I_p⊗u^T + u^T⊗I_p)(B⊗B)^T Pz22 vec_term
        T4_6 = (np.kron(I_p, u.T) + np.kron(u.T, I_p)) @ (kron_term.T @ Pz22 @ vec_term)
        # T4_7: (I_p⊗u^T + u^T⊗I_p)(B⊗B)^T Pz22^T vec_term
        T4_7 = (np.kron(I_p, u.T) + np.kron(u.T, I_p)) @ (kron_term.T @ Pz22.T @ vec_term)
        
        return T4_1 + T4_2 + T4_3 + T4_4 + T4_5 + T4_6 + T4_7

    def F(self, u, params):
        """
        Total gradient F(u) = 2 R u + term2 + term3 + term4.
        Here R is assumed provided in params.
        """
        R = params['R']
        t1 = 2 * R @ u
        t2 = self.term2(u, params)
        t3 = self.term3(u, params)
        t4 = self.term4(u, params)
        output = t1 + t2 + t3 + t4
        return output

    def approx_hessian(self,u, F_func, params, epsilon=1e-6):
        """
        Approximate Hessian H(u) = dF/du using finite differences.
        """
        p = self.p
        H = np.zeros((p, p, 1))
        f0 = F_func(u, params)
        for i in range(p):
            u_eps = u.copy() # shape (p,1)
            u_eps[i] += epsilon 
            f1 = F_func(u_eps, params)
            H[:, i, :] = ((f1 - f0) / epsilon)
        return H

    def newton_method(self, u0, params, epsilon=1e-6, max_iter=1000, verbose=True, plot=True):
        """
        Run Newton's method for k iterations to solve F(u)=0.
        """
        diff_list = []
        u = u0.copy() # initial guess for u
        for iteration in range(max_iter):
            # print(f"Iteration {iteration+1}")
            grad = self.F(u, params)
            Hessian = self.approx_hessian(u, self.F, params)
            # regularise
            H_reg = np.squeeze(Hessian) + np.eye(self.p) * 1e-6 # regularisation term
            H_reg = 0.5*(H_reg + H_reg.T)                 # force symmetry
            if not np.all(np.isfinite(H_reg)):        # NaN / Inf guard
                print(H_reg)
                raise ValueError("Hessian has non-finite entries")
            # delta = grad / Hessian
            delta = (np.linalg.pinv(H_reg) @ np.squeeze(grad)).reshape(-1,1) # shape (p,1)
            u = u - delta  # update u
            diff = np.linalg.norm(delta)
            diff_list.append(diff)
            if diff < epsilon:
                if verbose:
                    print(f"Converged at iteration {iteration+1} with diff {diff} < {epsilon}")
                break
        if plot:
            # plot convergence
            import matplotlib.pyplot as plt
            plt.plot(diff_list, label='Convergence')
            plt.xlabel('Iteration')
            plt.ylabel('Difference')
            plt.title('Convergence of Newton\'s Method')
            plt.legend()
            plt.grid()
            plt.savefig('LQG_QKF/newton_method_convergence_plot.png')
            if verbose:
                print('     last diff:', diff_list[-1])
            plt.show()
        
        return u

# main
if __name__ == "__main__":
    n = 3
    p = 2
    newton = NewtonSolver(n, p)
    u0 = np.random.rand(p, 1)  # initial guess for u
    
    # Pz(t+1) = [[Pz11(t+1), Pz12(t+1)], 
    #            [Pz21(t+1), Pz22(t+1)]]
    
    # z(t) = [z1(t), z2(t)]
    
    params = {
        'z1': np.random.rand(n, 1), # augmented state vector
        'z2': np.random.rand(n**2, 1), # augmented state vector
        'A': np.random.rand(n, n), # state transition matrix
        'B': np.random.rand(n, p), # control input matrix
        'Q': np.random.rand(n + n**2, n + n**2),  # cost matrix for state
        'R': np.random.rand(p, p),  # cost matrix for control inputs
        
        # to be figured out:
        'Pz11': np.random.rand(n, n), # cost-to-go matrix, element 1
        'Pz21': np.random.rand(n**2, n), # cost-to-go matrix, element 2
        'Pz12': np.random.rand(n, n**2), # cost-to-go matrix, element 3
        'Pz22': np.random.rand(n**2, n**2), # cost-to-go matrix, element 4
        'W': np.random.rand(n, n),  # covariance matrix for process noise w 
        # 'V': np.random.rand(n, n),   # covariance matrix for measurement noise v
    }
    
    k = 100  # number of iterations
    result = newton.newton_method(u0, params)
    print("Final result: u =", result)
    
    
# to be checked and tested:
#   - check if the expression for term2, term3, and term4 are correct
#   - check if the gradient and Hessian are correct
#   - check if the test result is reasonable