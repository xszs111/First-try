import numpy as np

def soft_thresholding_operator(x, lambda_):
    return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)

def bcd_lasso(A, b, lambda_, max_iter=1000, tol=1e-4):
    m, n = A.shape
    x = np.zeros((n, 1))
    residuals = []
    for iteration in range(max_iter):
        for j in range(n):
            r = b - np.dot(A, x) + A[:, j].reshape(-1, 1) * x[j]
            x[j] = soft_thresholding_operator(np.dot(A[:, j].T, r), lambda_) / np.dot(A[:, j].T, A[:, j])
        residuals.append(np.linalg.norm(A.dot(x) - b))
        if iteration > 0 and np.abs(residuals[-1] - residuals[-2]) < tol:
            break
    return x, residuals

# Set random seed for reproducibility
np.random.seed(2021)

# Generate random matrix A and vector b
A = np.random.rand(500, 100)
x_true = np.zeros((100, 1))
x_true[:5, 0] += np.array([i+1 for i in range(5)])
b = np.matmul(A, x_true) + np.random.randn(500, 1) * 0.1

# Set regularization parameter lambda
lambda_1 = 0.1
lambda_2=1
lambda_3=10

# Solve LASSO problem using BCD
x_hat1, residuals1 = bcd_lasso(A, b, lambda_1)
x_hat2, residuals2 = bcd_lasso(A, b, lambda_2)
x_hat3, residuals3 = bcd_lasso(A, b, lambda_3)
print(x_hat1,x_hat2,x_hat3)

# Plot residuals
import matplotlib.pyplot as plt
plt.plot(residuals1)
plt.plot(residuals2)
plt.plot(residuals3)
plt.xlabel('Iteration')
plt.ylabel('Residual')
plt.legend(['lam=0.1','lam=1','lam=10'])
plt.title('Convergence of BCD LASSO')
plt.show()