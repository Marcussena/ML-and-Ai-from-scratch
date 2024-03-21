import numpy as np

class NaiveBayes:
    def __init__(self):
        self.class_prior = None
        self.feature_likelihood = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # create empty arrays to store mean, var and prior probabilities of each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.class_prior = np.zeros(n_classes)

        # empty array to store the likelihood of each feature
        self.feature_likelihood = np.zeros((n_classes, n_features))
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            # populate the arrays with the mean and variance of each feature
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            # calculate prior log probabilities for each class
            self.class_prior[idx] = np.log(X_c.shape[0]/float(n_samples))

            # and calculate the likelihood of each sample given the class it belongs
            # self.feature_likelihood[idx, :] = np.array([self._gaussian(idx, x) for x in X_c])
            
        

    def predict_single(self, x):
        posteriors = []

        # calculate the posterior values for each class
        for idx, c in enumerate(self.classes):
            prior = self.class_prior[idx]
            # use the log of the likelihoods to optimize calculation
            likelihood = np.sum(np.log(self._gaussian(idx,x)))
            self.feature_likelihood[idx, :] = likelihood
            posterior = prior + likelihood
            posteriors.append(posterior)

        self.posteriors = posteriors

        # return the class with highest probability
        predicted_class = self.classes[np.argmax(posteriors)]
        return predicted_class

    def predict(self, X):
        # predict the class for each sample in the dataset
        y_pred = [self.predict_single(x) for x in X]
        return y_pred

    # auxiliar function to estimate the likelihoods according to gaussian distribution
    def _gaussian(self, class_id, x):
        mean = self._mean[class_id]
        var = self._var[class_id]

        pdf = (1/(np.sqrt(2 * np.pi * var)))*np.exp(-((x - mean) ** 2) / (2 * var))
        return pdf


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    # auxiliar function to calculate the accuracy of the model
    def accuracy(y_true, y_pred):
        accuracy = (np.sum(y_true == y_pred) / len(y_true))*100
        return accuracy
    
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)

    model = NaiveBayes()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(accuracy(y_test, y_pred))


