import numpy as np

class Layers:
    def __init__(self, numNodesIn, numNodesOut, activation='relu'):
        self.activation = activation
        self.w = np.random.randn(numNodesIn, numNodesOut)
        self.b = np.random.randn(1, numNodesOut)
        self.prev_input = None
        self.z = None
        self.a = None
        self.delta = None
        self.gamma = np.random.randn()
        self.beta = np.random.randn()
        self.n = None
        self.x_hat = None
        self.mean = None
        self.variance = None
    
    # Activation Functions
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    ##

    def batch_norm(self, x):
        self.mean = np.mean(x, axis=0)
        self.variance = np.var(x, axis=0)
        self.x_hat = (x - self.mean) / np.sqrt(self.variance + 1e-9)
        return self.gamma * self.x_hat + self.beta

    def forward(self, input):
        self.prev_input = input
        self.z = np.dot(input, self.w) + self.b
        if self.activation == 'relu':
            # self.n = self.batch_norm(self.z)
            self.a = self.relu(self.z)
        elif self.activation == 'softmax':
            self.a = self.softmax(self.z)
        return self.a
    
    def backward(self, input, learning_rate):
        if self.activation == 'relu':
            activation_derivative = self.relu_derivative(self.z)
        elif self.activation == 'softmax':
            activation_derivative = 1
        
        self.delta = input * activation_derivative
        previous_weights = self.w.copy()

        self.w -= learning_rate * np.dot(self.prev_input.T, self.delta) / input.shape[0]
        self.b -= learning_rate * np.sum(self.delta, axis=0, keepdims=True) / input.shape[0]
        
        return np.dot(self.delta, previous_weights.T)