import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(0)

# Generate correlated data
# Number of samples
n_samples = 100

# Create a covariance matrix for generating correlated data
correlation_matrix = np.array([
    [1.0, 0.8, 0.3, 0.0, 0.0],
    [0.8, 1.0, 0.4, 0.0, 0.0],
    [0.3, 0.4, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.5],
    [0.0, 0.0, 0.0, 0.5, 1.0]
])

# Generate the correlated data
mean = np.zeros(correlation_matrix.shape[0])
X_correlated = np.random.multivariate_normal(mean, correlation_matrix, size=n_samples)

# Compute the covariance matrix of the correlated data
covariance_matrix = np.cov(X_correlated, rowvar=False)

# Create a heatmap of the covariance matrix
plt.figure(figsize=(10, 8))
sns.heatmap(covariance_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Covariance Matrix Heatmap')
plt.xlabel('Features')
plt.ylabel('Features')
plt.show()
