import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LinearRegression:
    def __init__(self, method='closed-form', eta=0.01, n_iterations=400):
        # parameters initialization
        self.method = method
        self.eta = eta
        self.n_iterations = n_iterations
    
    # mean squared error function
    def MSE(self, X, y):
        m = len(y)
        predictions = self.predict(X)
        errors = predictions - y
        cost = (2 / m) * np.sum(errors ** 2)
        return cost
    
        # scale function
    def scale(self,X,y):
        scale_X = (X - np.mean(X))/np.std(X)
        scale_y = (y - np.mean(y))/np.std(y)

        return scale_X, scale_y
    
    # unscale function
    def unscale(self,w,q):
        a = w*(np.std(self.y)/np.std(self.X))
        b = q*np.std(self.y) + np.mean(self.y) - w*np.mean(self.X)*np.std(self.y)/np.std(self.X)

        return a,b
    

    # closed-form solution
    def closed_form(self,X,y):
        X_bar = np.mean(X)
        y_bar = np.mean(y)
        
        a = np.sum([(xi - X_bar)*(yi - y_bar) for xi,yi in zip(X,y)])
        b = np.sum([(xi - X_bar)**2 for xi in X])

        alpha = a/b
        beta = y_bar - alpha*X_bar

        return alpha, beta
    
    # gradient descent solution
    def gradient_descent(self,X, y):
        self.X = X
        self.y = y

        m = len(y)

        X,y = self.scale(X,y)

        # initialize the parameters with random small values
        self.alpha = np.random.rand(1)
        self.beta = np.random.rand(1)

        mse_error = []
        for _ in range(self.n_iterations):
            error = (self.alpha*X + self.beta) - y
            alpha_grad = (1/m)*np.sum(X*error)
            beta_grad = (1/m)*np.sum(error)
            
            self.alpha = self.alpha - self.eta*alpha_grad
            self.beta = self.beta - self.eta*beta_grad
            mse_error.append(self.MSE(X,y))

        self.mse_error = mse_error
        alpha, beta = self.unscale(self.alpha, self.beta)
        return alpha, beta
    
    def fit(self,X,y):
        
        if self.method == 'closed-form':
            self.alpha, self.beta = self.closed_form(X,y)
        elif self.method=='gradient-descent':
            self.alpha, self.beta = self.gradient_descent(X,y)
        
    # predict function for a dataset
    def predict(self, X):
        return self.alpha*X + self.beta
    

    

if __name__ == '__main__':

    data = pd.read_csv("archive/tvmarketing.csv")
    TV = np.array(data.TV, dtype=np.float64)
    sales = np.array(data.Sales, dtype=np.float64)

    model = LinearRegression(method='gradient-descent')
    model.fit(TV,sales)
    a,b = model.alpha, model.beta

    sales_pred = model.predict(TV)
    
    plt.title("Mean Squred error over epochs")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.plot(model.mse_error)
    
    plt.title("Predicted sales x advertisement on TV")
    plt.scatter(TV,sales)
    plt.plot(TV, sales_pred, "r-")
    plt.xlabel("TV advertisement")
    plt.ylabel("Sales")
    plt.show()



