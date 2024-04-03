import numpy as np

class SVM:
    # Basic initialization function
    def __init__(self, lr = 0.001, lambda_param = 0.01, n_iter = 1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iter = n_iter
        self.w = None
        self.b = None

    # Function to initializate the weight vector and the bias term with random small values
    def initialize_weights(self, X):
        n_features = X.shape[1]
        self.w = np.random.rand(n_features)
        self.b = np.random.rand()

    # Function to label the clas of each sample as 1 or -1 
    def get_label(self, y):
        y_ = np.where(y<=0, -1, 1)
        return y_
    
    # Fit function to train from data using gradient descent
    def fit(self, X, y):
        self.initialize_weights(X)
        labels = self.get_label(y)

        for _ in range(self.n_iter):
            for idx, xi in enumerate(X):
                condition = labels[idx]*(np.dot(xi,self.w) + self.b) >=1
                if condition:
                    self.w -= self.lr*(2*self.lambda_param*self.w)
                else:
                    self.w -= self.lr*(2*self.lambda_param*self.w - np.dot(xi,labels[idx]))
                    self.b -= self.lr*labels[idx]

    # Predict function to perform classification over new data points
    def predict(self, X):
        # predict the label for a new data point assuming a linear function
        pred = np.dot(X, self.w) + self.b
        return np.sign(pred)
    
# Test case
if __name__ == '__main__':
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import matplotlib.pyplot as plt
    import pandas as pd

    # Load the Iris dataset from sklearn
    iris_data = datasets.load_iris()
    data = iris_data.data
    target = iris_data.target

    X = data[:100,[0,2]]
    y = target[target<2]
    y = np.where(y==0, -1, 1)

    # split data in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # create dataframes to create plots
    iris_df_train = pd.DataFrame(X_train, columns=[iris_data.feature_names[0], iris_data.feature_names[2]])
    iris_df_train['species'] = y_train

    iris_df_test = pd.DataFrame(X_test, columns=[iris_data.feature_names[0], iris_data.feature_names[2]])
    iris_df_test['species'] = y_test

    # Create a scatter plot with different symbols for each target value
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(iris_df_train['sepal length (cm)'], iris_df_train['petal length (cm)'],
                      c=iris_df_train['species'], marker='o', s=50)
    
    # Legend
    legend_elements = scatter.legend_elements()
    legend = plt.legend(legend_elements[0], legend_elements[1], title="Species")
    plt.title('Scatter Plot of Iris Dataset')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')

    # show the plot of training data
    plt.show()

    clf = SVM()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy*100

    print("SVM classification accuracy", accuracy(y_test, predictions))
    print(clf.w, clf.b)

    def plot_svm():
        # helper function to get the values for each hyperplane
        def get_values(x,w,b,offset):
            return (-w[0]*x - b + offset)/w[1]
        
        xmin = np.min(X[:,0])
        xmax = np.max(X[:,0])

        # Optimal hyperplane values
        x1_optimal = get_values(xmin, clf.w, clf.b, 0)
        x2_optimal = get_values(xmax, clf.w, clf.b, 0)

        # Support vectors values
        x1_s1 = get_values(xmin, clf.w, clf.b, 1)
        x2_s1 = get_values(xmax, clf.w, clf.b, 1)

        x1_s2 = get_values(xmin, clf.w, clf.b, -1)
        x2_s2 = get_values(xmax, clf.w, clf.b, -1)

        # plot the lines
        plt.figure(figsize=(10, 8))
        plt.scatter(iris_df_train['sepal length (cm)'], iris_df_train['petal length (cm)'],
                      c=iris_df_train['species'], marker='o', s=50)
        
        plt.scatter(iris_df_test['sepal length (cm)'], iris_df_test['petal length (cm)'],
                      c=iris_df_test['species'], marker='x', s=50)
    
        plt.plot([xmin, xmax], [x1_optimal, x2_optimal], "y--")
        plt.plot([xmin, xmax], [x1_s1, x2_s1], "k")
        plt.plot([xmin, xmax], [x1_s2, x2_s2], "k")

        # Legend
        legend_elements = scatter.legend_elements()
        legend = plt.legend(legend_elements[0], legend_elements[1], title="Species")
        plt.title('Scatter Plot of Iris Dataset')
        plt.xlabel('Sepal Length (cm)')
        plt.ylabel('Petal Length (cm)')

        plt.show()

    plot_svm()

