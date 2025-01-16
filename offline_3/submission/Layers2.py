import numpy as np
from AdamOptimizer import AdamOptimizer

class Layers:
    def __init__(self, numNodesIn, numNodesOut, drop_out_prob = None, activation='relu',  learning_rate=0.0001, beta1=0.9, beta2=0.999):
        self.activation = activation
        self.w = np.random.randn(numNodesIn, numNodesOut) * np.sqrt(2. / numNodesIn)
        self.b = np.zeros((1, numNodesOut))  
        self.drop_out_prob = drop_out_prob
        self.dropout_mask = None
        self.prev_input = None
        self.z = None
        self.a = None
        self.delta = None
        self.gamma = np.ones((1, numNodesOut))  
        self.beta = np.zeros((1, numNodesOut)) 
        self.n = None
        self.x_hat = None
        self.mean = None
        self.variance = None
        self.out = None
        self.cache = None
        self.training = True  
        self.running_mean = np.zeros((1, numNodesOut))
        self.running_var = np.zeros((1, numNodesOut))

        self.w_optimizer = AdamOptimizer(learning_rate, beta1, beta2)
        self.b_optimizer = AdamOptimizer(learning_rate, beta1, beta2)
        self.gamma_optimizer = AdamOptimizer(learning_rate, beta1, beta2)
        self.beta_optimizer = AdamOptimizer(learning_rate, beta1, beta2)

    # Activation Functions
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def set_training_mode(self, training=True):
        self.training = training

    def batchnorm_forward(self, z):
        N, D = z.shape
        if self.training:
            mu = np.mean(z, axis=0, keepdims=True)
            var = np.var(z, axis=0, keepdims=True)
            xhat = (z - mu) / np.sqrt(var + 1e-8)
            self.out = self.gamma * xhat + self.beta
            momentum = 0.9
            self.running_mean = momentum * self.running_mean + (1 - momentum) * mu
            self.running_var = momentum * self.running_var + (1 - momentum) * var
            self.cache = (xhat, z - mu, 1. / np.sqrt(var + 1e-8), np.sqrt(var + 1e-8), var)
        else:
            xhat = (z - self.running_mean) / np.sqrt(self.running_var + 1e-8)
            self.out = self.gamma * xhat + self.beta
        return self.out
    
    def dropout(self, x):
        if self.drop_out_prob is not None and self.training:
            self.dropout_mask = (np.random.rand(1, x.shape[1]) > self.drop_out_prob).astype(float)
            return x * self.dropout_mask / (1 - self.drop_out_prob)
        else:
            return x

    def forward(self, input):
        self.prev_input = input  
        self.z = np.dot(input, self.w) + self.b  
        if self.activation == 'relu':
            self.n = self.batchnorm_forward(self.z)  
            self.a = self.dropout(self.relu(self.n))  
        elif self.activation == 'softmax':
            self.a = self.softmax(self.z)  
        return self.a
    
    def batchnorm_backward(self, dout):
        xhat, xmu, ivar, sqrtvar, var = self.cache
        N, D = dout.shape

        dbeta = np.sum(dout, axis=0, keepdims=True)
        dgamma = np.sum(dout * xhat, axis=0, keepdims=True)
        dxhat = dout * self.gamma
        dvar = np.sum(dxhat * xmu * -0.5 * (var + 1e-8) ** (-1.5), axis=0, keepdims=True)
        dmu = np.sum(dxhat * -ivar, axis=0, keepdims=True) + dvar * np.mean(-2. * xmu, axis=0, keepdims=True)
        dx = dxhat * ivar + dvar * 2 * xmu / N + dmu / N

        return dx, dgamma, dbeta

    def backward(self, upstream_grad):
        if self.activation == 'relu':
            activation_derivative = self.relu_derivative(self.n)

            self.delta = upstream_grad * activation_derivative 
            if self.drop_out_prob is not None and self.training:
                self.delta *= self.dropout_mask 

            dz, dgamma, dbeta = self.batchnorm_backward(self.delta) 

           
            gradient_w = np.dot(self.prev_input.T, dz) / self.prev_input.shape[0]  
            gradient_b = np.sum(dz, axis=0, keepdims=True) / self.prev_input.shape[0]  

            self.w_optimizer.update(self.w, gradient_w)
            self.b_optimizer.update(self.b, gradient_b)

            self.gamma_optimizer.update(self.gamma, dgamma / self.prev_input.shape[0])
            self.beta_optimizer.update(self.beta, dbeta / self.prev_input.shape[0])

            return np.dot(dz, self.w.T) 

        elif self.activation == 'softmax':
            self.delta = upstream_grad 

            gradient_w = np.dot(self.prev_input.T, self.delta) / self.prev_input.shape[0]  
            gradient_b = np.sum(self.delta, axis=0, keepdims=True) / self.prev_input.shape[0] 

            self.w_optimizer.update(self.w, gradient_w)
            self.b_optimizer.update(self.b, gradient_b)

            return np.dot(self.delta, self.w.T)  