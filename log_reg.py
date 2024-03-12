import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random


class LogisticRegression:
    # init method to set parameters
    def __init__(self, lr=0.01, epochs=10000):
        self.lr = lr
        self.epochs = epochs
        self.alpha = 0
        self.beta = 0

    # auxiliar function to calculate the sigmoid
    def sigmoid(self, z):
        res = 1 / (1 + np.exp(-z))
        return res

    # function to calculate the cost function regarding the actual y and y predicted
    def cost(self, y, y_pred):
        m = len(y)
        cost = (-1 / m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return cost

    # fit function to find the optimal parameters
    def fit(self, X, y):
        m = len(y)

        for epoch in range(self.epochs):
            # use linear equation to get inputs for sigmoid
            y_pred = self.sigmoid(self.alpha * X + self.beta)

            # calculate the gradients for each parameter
            error = y_pred - y
            alpha_grad = (1 / m) * np.sum(error * X)
            beta_grad = (1 / m) * np.sum(error)

            # update the parameters with gradient descent
            self.alpha -= self.lr * alpha_grad
            self.beta -= self.lr * beta_grad

            # check the value of the cost function every 100 iterations
            if epoch % 100 == 0:
                cost = self.cost(y, y_pred)
                print(f"Epoch {epoch} - cost: {cost:.4f}")

    def predict(self, X):
        y_pred = self.sigmoid(self.alpha * X + self.beta)

        # return 0 if y_pred < 0.5 and 1 otherwise
        return (y_pred > 0.5).astype(int)
        # return y_pred

    # accuracy function that calculates how many predictions are right
    def accuracy(self, y_pred, y_test):
        # Check if the input lists have the same length
        if len(y_test) != len(y_pred):
            raise ValueError("Input lists must have the same length.")

        # Count the number of correct predictions
        correct_predictions = sum(1 for test, pred in zip(y_test, y_pred) if test == pred)

        # Calculate accuracy
        accuracy = correct_predictions / len(y_test) * 100.0

        return accuracy


if __name__ == '__main__':
    def scale(X, y):
        scale_X = (X - np.mean(X)) / np.std(X)
        scale_y = (y - np.mean(y)) / np.std(y)

        return scale_X, scale_y


    # auxiliar function to split dataset
    def train_test_split(data, labels, test_size=0.2):
        # Check if data and labels have the same length
        if len(data) != len(labels):
            raise ValueError("Data and labels must have the same length.")

        # Shuffle indices
        indices = list(range(len(data)))
        random.shuffle(indices)

        # Calculate the number of samples for the test set
        test_size = int(len(data) * test_size)

        # Split the data and labels
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]

        X_train = np.array([data[i] for i in train_indices])
        X_test = np.array([data[i] for i in test_indices])
        y_train = np.array([labels[i] for i in train_indices])
        y_test = np.array([labels[i] for i in test_indices])

        return X_train, X_test, y_train, y_test


    df = pd.read_csv('Default.csv')
    balance = np.array(df.balance)
    default = np.array(df.default.map({"No": 0, "Yes": 1}))

    # split data in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(balance, default, test_size=0.25)

    balance_train, _ = scale(X_train, y_train)
    balance_test, _ = scale(X_test, y_test)

    model = LogisticRegression()
    model.fit(balance_train, y_train)

    print(model.alpha, model.beta)

    y_pred = model.predict(balance_test)
    plt.figure(figsize=(12, 6))
    plt.title("Logistic regression")
    plt.xlabel("scaled balance")
    plt.ylabel("Default")
    plt.scatter(balance_test, y_test)
    plt.scatter(balance_test, y_pred)
    plt.legend(["test data", "predicted data"])

    plt.show()

    acc = model.accuracy(y_pred, y_test)
    print(acc)
