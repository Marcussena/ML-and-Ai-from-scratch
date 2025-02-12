import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression:
    """
    A Linear Regression model with multiple training methods:
    - Closed-Form Solution
    - Gradient Descent
    - Stochastic Gradient Descent (SGD)
    - Mini-Batch Gradient Descent
    """
    def __init__(self):
        """Initializes the model parameters (weights and bias) as None."""
        self.weights = None
        self.bias = None

    def _initialize_params(self, n_features):
        """Initializes weights to zeros and bias to zero."""
        self.weights = np.zeros(n_features)
        self.bias = 0.0

    def predict(self, X):
        """
        Predicts target values based on input features X.
        
        Parameters:
        X (ndarray): Feature matrix.
        
        Returns:
        ndarray: Predicted values.
        """
        return np.dot(X, self.weights) + self.bias
    
    def fit_closed_form(self, X, y):
        """
        Trains the model using the Closed-Form solution (Normal Equation).
        
        Parameters:
        X (ndarray): Feature matrix.
        y (ndarray): Target values.
        """
        # Add a bias column (column of ones) to the feature matrix
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # Compute the optimal parameters using the Normal Equation
        params = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        self.bias = params[0]
        self.weights = params[1:]

    def fit_gradient_descent(self, X, y, eta=0.1, epochs=1000, callback=False):
        """
        Trains the model using Batch Gradient Descent.
        
        Parameters:
        X (ndarray): Feature matrix.
        y (ndarray): Target values.
        eta (float): Learning rate.
        epochs (int): Number of iterations.
        callback (function, optional): Function to track training progress.
        """
        self._initialize_params(X.shape[1])
        n_samples = X.shape[0]

        for i in range(epochs):
            y_pred = self.predict(X)
            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)

            self.weights -= eta * dw
            self.bias -= eta * db
            if callback:
                d = self.evaluate(X,y)
                print(f"Epoch: {i} - {d}")

    def fit_SGD(self, X, y, epochs=10000, callback=False):
        """
        Trains the model using Stochastic Gradient Descent (SGD).
        
        Parameters:
        X (ndarray): Feature matrix.
        y (ndarray): Target values.
        epochs (int): Number of epochs.
        callback (function, optional): Function to track training progress.
        """
        self._initialize_params(X.shape[1])
        t0, t1 = 5, 50  # Learning schedule parameters
        n_samples = X.shape[0]

        def learning_schedule(t):
            return t0 / (t + t1)
        
        for i in range(epochs):
            for sample in range(n_samples):
                # Select a random training sample
                idx = np.random.randint(n_samples)
                x_i = X[idx:idx + 1]
                y_i = y[idx:idx + 1]

                y_pred = self.predict(x_i)
                dw = 2 * np.dot(x_i.T, (y_pred - y_i))
                db = 2 * np.sum(y_pred - y_i)

                eta = learning_schedule(i * n_samples + sample)
                self.weights -= eta * dw
                self.bias -= eta * db
            if callback:
                d = self.evaluate(X,y)
                print(f"Epoch: {i} - {d}")
        
    def fit_mini_batch(self, X, y, eta=0.1, epochs=10000, batch_size=40, callback=False):
        """
        Trains the model using Mini-Batch Gradient Descent.
        
        Parameters:
        X (ndarray): Feature matrix.
        y (ndarray): Target values.
        eta (float): Learning rate.
        epochs (int): Number of epochs.
        batch_size (int): Number of samples per batch.
        callback (function, optional): Function to track training progress.
        """
        self._initialize_params(X.shape[1])
        n_samples = X.shape[0]

        for i in range(epochs):
            # Select a random batch of samples
            batch = np.random.choice(n_samples, batch_size, replace=False)
            X_batch = X[batch]
            y_batch = y[batch]

            y_pred = self.predict(X_batch)

            dw = (2 / batch_size) * np.dot(X_batch.T, (y_pred - y_batch))
            db = (2 / batch_size) * np.sum(y_pred - y_batch)

            self.weights -= eta * dw
            self.bias -= eta * db
            if callback:
                d = self.evaluate(X,y)
                print(f"Epoch: {i} - {d}")

    def evaluate(self, X, y):
        """
        Evaluates the model using common regression metrics.
        
        Parameters:
        X (ndarray): Feature matrix.
        y (ndarray): Target values.
        
        Returns:
        dict: Dictionary containing MAE, MSE, RMSE, and R2 scores.
        """
        
        y_pred = self.predict(X)

        # Compute evaluation metrics
        mae = np.mean(np.abs(y - y_pred)).item()  # Mean Absolute Error
        mse = np.mean((y - y_pred) ** 2).item()  # Mean Squared Error
        rmse = np.sqrt(mse).item()  # Root Mean Squared Error
        r2 = (1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)).item()  # R-squared

        # Store metrics in a dictionary
        metrics_d = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        }
        
        return metrics_d



    

if __name__ == '__main__':
    from sklearn.datasets import load_diabetes

    # load the diabetes dataset
    data, target = load_diabetes(return_X_y=True)
    
    # Create train and test datasets
    size = len(data)
    split = int(0.75 * size)
    
    X_train = data[0:split, [2,3,4,5]]
    X_test = data[split:, [2,3,4,5]]

    y_train = target[0:split]
    y_test = target[split:]


    # Initialize the model
    model = LinearRegression()

    # Closed-form example
    model.fit_closed_form(X_train, y_train)

    # Gradient Descent example
    model.fit_gradient_descent(X_train, y_train)

    # Stochastic Gradient Descent example
    model.fit_SGD(X_train, y_train)

    # Mini-batch example
    model.fit_mini_batch(X_train, y_train)

    # evaluate on test data
    d = model.evaluate(X_test, y_test)
    print(d)





    
