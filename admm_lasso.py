import numpy as np
import matplotlib.pyplot as plt


def soft_threshold(v,k):
    n = np.shape(v)[0]
    v_new = np.zeros([n,1])
    
    for i in range(n):
        if v[i,0] >= k:
            v_new[i,0] = v[i,0] - k
        elif v[i,0] <= -k:
            v_new[i,0] = v[i,0] + k
        else:
            v_new[i,0] = 0
    return v_new
    
def factor(A, rho):
    
    m, n = A.shape
    if m >= n:
        L = np.linalg.cholesky(np.dot(A.T, A) + rho*np.eye(n))
    else:
        L = np.linalg.cholesky(np.eye(m) + 1/rho*(np.dot(A, A.T)))

    L = L
    U = L.T.conj()
    return L, U
    

def plot(err,err_r):
    n = np.shape(err)[0]
    n = np.arange(n)
    plt.figure(1)
    plt.clf()

    plt.subplot(211)

    plt.plot(n, err, 'r')
    plt.ylabel('absolute error')

    plt.subplot(212)
    plt.plot(n, err_r, 'r')
    plt.hlines(0,0,np.shape(n)[0], colors = 'orange', linestyles = 'dashed', zorder = 10)
    plt.ylabel('relative error')
    plt.xlabel('step')

    plt.show()
    
    
def admm_lasso(A, b, x0, alpha = 1.0, rho = 1.0, lambda_ = 1.0, N_iter = 100):
    N_sample, Nx = np.shape(A)
    x = np.zeros([Nx,1])
    z = np.zeros([Nx,1])
    u = np.zeros([Nx,1])
    
    err = []    #  Absolute error
    err_r = []    #  Relative error
    
    # cache the factorization
    L, U = factor(A, rho)
    Atb = A.T @ b
    
    for i in range(N_iter):

        q = Atb + rho*(z - u)  # temporary value
        if( N_sample >= Nx ):    # if skinny, using direct method with the help of Cholesky decomposition.
            x = np.linalg.solve(U, np.linalg.solve(L, q))
        else:            # if fat, using matrix inversion lemma.
            x = q/rho - A.T @ np.linalg.solve(U, np.linalg.solve(L, A @ q))/rho**2
        zold = z
        x_hat = alpha*x + (1 - alpha)*zold

    #     x = np.linalg.inv(A.T @ A + rho * np.eye(p)) @ (A.T @ b + rho* (z - u))
        z = soft_threshold(x_hat + u, lambda_/rho)
        u = u + x_hat - z

        err += [np.linalg.norm(x-x0)]
        err_r += [np.linalg.norm(x-z)]   
        
    return err,err_r

if __name__ == '__main__':
    N_sample = 5000    #  Number of samples
    Nx = 1500    #  Number of regressor
    A = np.random.randn(N_sample,Nx)
    A = np.dot(A, sparse.spdiags(1/np.sqrt(sum(np.multiply(A, A))), 0, Nx, Nx).todense())  # make it symmetric and normalize column

    x0 = np.random.randn(Nx,1)
    indices = np.random.choice(np.arange(x0.size), replace=False,
                               size=int(x0.size * 0.8))
    x0[indices] = 0

    b = A @ x0 + np.sqrt(0.001)*np.random.randn(N_sample, 1)

    lambda_max = np.linalg.norm(np.dot(A.T, b), np.inf)
    lambda_ = 0.1*lambda_max

    rho = 1.0
    alpha = 1.0    #  Over-relaxation: select alpha in [1.5, 1.8] to improve covergence.

    N_iter = 100;

    err, err_r = admm_lasso(A, b, x0, alpha = alpha, rho = rho, lambda_ = lambda_, N_iter = N_iter)
    plot(err, err_r)