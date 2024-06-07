import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(0)
X = np.random.randn(100, 2) + np.array([5, 10])

# Plot the original data
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], color='blue', alpha=0.5)
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)

# Subtract the mean from the data
mean = np.mean(X, axis=0)
X_centered = X - mean

# Plot the centered data
plt.subplot(1, 2, 2)
plt.scatter(X_centered[:, 0], X_centered[:, 1], color='green', alpha=0.5)
plt.title('Centered Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)

# Show the plots
plt.tight_layout()
plt.show()
