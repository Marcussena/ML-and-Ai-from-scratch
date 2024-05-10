import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature=feature
        self.threshold=threshold
        self.left=left
        self.right=right
        self.value=value

    # if the node has a value it is a leaf node
    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples=2, max_depth=2):
        self.min_samples = min_samples
        self.max_depth = max_depth

    # Helper function to calculate the entropy of a partition
    def entropy(self, y):
        entropy = 0
    
        # get the unique target values
        labels = np.unique(y)

        for label in labels:
            # get all the elements with the current value label
            label_elements = y[y==label]

            # calculate the label probability
            pl = len(label_elements)/len(y)

            # then calculate the entropy
            entropy += -pl*np.log2(pl)

        return entropy
    
    # splits the data into left and right nodes
    def split_data(self, dataset, feature, threshold):
    
        # initialize empty lists to store splitted data
        left_data = []
        right_data = []

        for idx,row in enumerate(dataset):
            if row[feature] <= threshold:
                left_data.append(idx)
            else:
                right_data.append(idx)

        # turn left and right data into arrays
        left_idx = np.array(left_data)
        right_idx = np.array(right_data)

        # return the indexes of left and right nodes
        return left_idx, right_idx
    
    # Helper function to calculate the information gain of a split
    def information_gain(self, target, left, right):
        parent_entropy = self.entropy(target)

        left_entropy, right_entropy = self.entropy(target[left]), self.entropy(target[right]) 

        # calculate the whieghts of each node
        left_weight = len(left)/len(target)
        right_weight = len(right)/len(target)

        # calculate the information gain
        info_gain = parent_entropy - (left_weight*left_entropy + right_weight*right_entropy)
        return info_gain
    
    # method that returns the feature and threshold values that splits the data with the larger information gain
    def best_split(self, dataset, target, num_features):
        best_gain = -1

        for feat_idx in range(num_features):

            # get the column of the current feature
            data_column = dataset[:,feat_idx]

            # get a list of each unique value of the column to use as threshold
            threshold = np.unique(data_column)

            # perform a split and calculate the information gain for each threshold value
            for thr in threshold:
                left, right = self.split_data(dataset, feat_idx, thr)

                # make sure that each split have samples
                if len(left) and len(right):
                    gain = self.information_gain(target, left, right)

                    if gain > best_gain:
                        best_gain = gain
                        split_idx = feat_idx
                        split_threshold = thr

        return split_idx, split_threshold, gain
    
    # Set the value of a leaf node based on the most common class label present
    def leaf_value(self, y):
        y = list(y)

        most_common = max(y, key=y.count)

        return most_common
    
    # function to split the data until stop conditions are achieved
    # Stopping rules:
    # Minimum number of samples in a node
    # When the number of labels in a node equals 1 (100% purity)
    # Maximum depth of the tree
    def grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # check stopping criteria
        if (n_samples < self.min_samples or n_labels == 1 or depth >= self.max_depth):
            node_value = self.leaf_value(y)
            return Node(value=node_value)

        # use the best split
        best_idx, best_thresh, best_gain = self.best_split(X, y, n_features)

        # creating the nodes
        if best_gain:
            left_idx, right_idx = self.split_data(X, best_idx, best_thresh)
            left_node = self.grow_tree(X[left_idx, :], y[left_idx], depth=depth+1)
            right_node = self.grow_tree(X[right_idx, :], y[right_idx], depth=depth+1)
            return Node(best_idx, best_thresh, left_node, right_node)

        
    def fit(self, X, y):
        self.root = self.grow_tree(X,y)

    def predict_value(self, x, node):
        # traverse the tree to find the value for a sample x
        if node.is_leaf():
            return node.value
        
        feat = node.feature
        if x[feat] <= node.threshold:
            return self.predict_value(x, node.left)
        return self.predict_value(x, node.right)
        
    # predict the leafs values for each sample on X
    def predict(self, X):
        pred_values = [self.predict_value(x, self.root) for x in X]
        return np.array(pred_values)



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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=50)

    model = DecisionTree(min_samples=2, max_depth=3)
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    acc = accuracy(y_test, y_pred)

    print(acc)


