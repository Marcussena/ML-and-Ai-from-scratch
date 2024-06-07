import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import random

class KNN:
    def __init__(self, k, task = 'classification'):
        self.k = k
        self.task = task

    # euclidean distance helper function
    def distance(self, a, b):
        # implements the euclidean distance between two points
        dist = np.sqrt(np.sum((a-b)**2))
        return dist
    
    # fit function
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict_point(self, point):
        # distance from each point to train data
        distances = [self.distance(point, x) for x in self.X_train]

        # find the k closest points
        indices = np.argsort(distances)[:self.k]

        # get the predicted label based on the task 
        labels = [self.y_train[ind] for ind in indices]

        if self.task == 'regression':
            return np.average(labels)
        elif self.task == 'classification':
            return np.bincount(labels).argmax()

    # predict function for test dataset    
    def predict(self, X_test):
        y_pred = [self.predict_point(x) for x in X_test]
        return np.array(y_pred)
    
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


# auxiliar function to split dataset
def train_test_split(data, labels, test_size = 0.2):
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

# Load the Iris dataset from sklearn
iris_data = datasets.load_iris()
data = iris_data.data
target = iris_data.target

# split data in train and test sets
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target)

# create dataframes to create plots
iris_df_train = pd.DataFrame(X_train, columns=iris_data.feature_names)
iris_df_train['species'] = y_train

iris_df_test = pd.DataFrame(X_test, columns=iris_data.feature_names)
iris_df_test['species'] = y_test

# Create a scatter plot with different symbols for each target value
plt.figure(figsize=(16, 10))
scatter = plt.scatter(iris_df_train['sepal length (cm)'], iris_df_train['sepal width (cm)'],
                      c=iris_df_train['species'], marker='o', s=50)

# plot test points
# plt.scatter(iris_df_test['sepal length (cm)'], iris_df_test['sepal width (cm)'],
                      # c='red', marker='x', s=50)

# Legend
legend_elements = scatter.legend_elements()
legend = plt.legend(legend_elements[0], legend_elements[1], title="Species")
plt.title('Scatter Plot of Iris Dataset')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')

# show the plot
# plt.show()

model = KNN(k=3, task='classification')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = model.accuracy(y_pred, y_test)
print(acc)

iris_df_pred = pd.DataFrame(X_test, columns=iris_data.feature_names)
iris_df_pred['species'] = y_pred


# Create a second scatter plot for predicted values
plt.figure(figsize=(16, 10))
scatter2 = plt.scatter(iris_df_pred['sepal length (cm)'], iris_df_pred['sepal width (cm)'],
                      c=iris_df_pred['species'], marker='o', s=50)

# Legend
legend_elements = scatter2.legend_elements()
legend = plt.legend(legend_elements[0], legend_elements[1], title="Species")
plt.title('Scatter Plot of Iris Dataset')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')

# show the plot
# plt.show()