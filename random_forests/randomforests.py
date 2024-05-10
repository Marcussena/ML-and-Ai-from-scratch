import numpy as np
from decisiontree import DecisionTree

class RandomForest:
    # constructor method for hyperparameter initialization
    def __init__(self, min_samples=2, max_depth=2, n_trees=10, n_features=None):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.n_trees = n_trees
        self.n_features = n_features
        self.trees = []

    
    # method for bootstrapping samples
    def bootastrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)

        return X[idxs], y[idxs]
    
    # method for training each tree in the ensemble using a random subset of parameters
    def fit(self, X, y):

        for _ in range(self.n_trees):
            tree = DecisionTree(min_samples=self.min_samples, max_depth= self.max_depth)
            X_sample, y_sample = self.bootastrap_sample(X,y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    
    # method to calculate the predicted value by majority vote
    def tree_value(self, y):
        y = list(y)

        most_common = max(y, key=y.count)

        return most_common
    
    def predict(self, X):
        pred_values = np.array([tree.predict(X) for tree in self.trees])
        true_preds = np.swapaxes(pred_values, 0, 1)
        predictions = np.array([self.tree_value(pred) for pred in true_preds])
        return predictions
    

if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    # auxiliar function to calculate the accuracy of the model
    def accuracy(y_true, y_pred):
        accuracy = (np.sum(y_true == y_pred) / len(y_true))*100
        return accuracy
    
    breast_cancer = load_breast_cancer()
    X = breast_cancer.data
    y = breast_cancer.target

    # split the data into rain and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)

    model = RandomForest(max_depth=10, n_trees=10, n_features=15)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy(y_test, y_pred)

    print(acc)



    