from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from linear_regression import LinearRegression
import numpy as np

# load the diabetes dataset
data, target = load_diabetes(return_X_y=True)

# Scaling
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
mean, std = scaler.mean_, scaler.scale_

# Create train and test datasets
size = len(data_scaled)
split = int(0.75 * size)
    
X_train = data_scaled[0:split]
X_test = data_scaled[split:]

y_train = target[0:split]
y_test = target[split:]

# Initialize the model
model = LinearRegression()

# Closed-form example
# model.fit_closed_form(X_train, y_train)
# d = model.evaluate(X_test, y_test)
# print(d)

# Gradient Descent example
model.fit_gradient_descent(X_train, y_train)
metrics = model.evaluate(X_test, y_test)
print(metrics)

# Stochastic Gradient Descent example
# model.fit_SGD(X_train, y_train, callback=True)

