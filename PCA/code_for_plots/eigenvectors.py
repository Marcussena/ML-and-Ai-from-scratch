import numpy as np
import matplotlib.pyplot as plt

# Generate a 2D dataset
np.random.seed(0)
mean = [2, 3]
cov = [[3, 1], [1, 2]]  # covariance matrix
X = np.random.multivariate_normal(mean, cov, 100)

# Standardize the dataset
X_centered = X - np.mean(X, axis=0)

# Compute the covariance matrix
cov_matrix = np.cov(X_centered, rowvar=False)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Plot the original data
plt.figure(figsize=(12, 8))
plt.scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.5, label='Data points')

# Plot the eigenvectors
for i in range(len(eigenvalues)):
    eigenvector = eigenvectors[:, i]
    start_point = np.mean(X_centered, axis=0)
    end_point = start_point + eigenvector * np.sqrt(eigenvalues[i]) * 3  # Scale by sqrt of eigenvalue for better visualization
    plt.arrow(start_point[0], start_point[1], end_point[0]-start_point[0], end_point[1]-start_point[1], 
              head_width=0.2, head_length=0.3, fc='red', ec='red', label=f'Eigenvector {i+1}')

plt.title('Data points and Eigenvectors')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
