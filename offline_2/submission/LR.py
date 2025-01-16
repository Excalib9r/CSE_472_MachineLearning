import numpy as np

class LogisticRegression:
    def __init__ (self, learning_rate=0.0001, n_iter=1000, tol=1e-6):
        self.lr = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.tol = tol
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def update_weights(self, X, y):
        y_pred = self.sigmoid(np.dot(X, self.weights))
        error = y - y_pred
        gradient = np.dot(X.T, error)
        self.weights += self.lr * gradient

    def train(self, X, y):
        n_samples, n_features = X.shape
        X = np.hstack((np.ones((n_samples, 1)), X))
        self.weights = np.zeros(n_features + 1)

        for _ in range(self.n_iter):
            previous_weights = self.weights.copy()
            self.update_weights(X, y)
            weight_change = np.linalg.norm(self.weights - previous_weights)
            if weight_change < self.tol:
                break
            
    def predict(self, X, final=True):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        if final:
            return np.round(self.sigmoid(np.dot(X, self.weights)))
        return self.sigmoid(np.dot(X, self.weights))
    

