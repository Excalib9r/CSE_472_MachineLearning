import numpy as np
import pickle
from AdamOptimizer import AdamOptimizer  # Ensure the correct import path

class Layers:
    def __init__(self, numNodesIn, numNodesOut, drop_out_prob=None, activation='relu',
                 learning_rate=0.0001, beta1=0.9, beta2=0.999):
        """
        Initializes the Layers class with Adam optimizer.

        Parameters:
            numNodesIn (int): Number of input nodes.
            numNodesOut (int): Number of output nodes.
            drop_out_prob (float, optional): Dropout probability.
            activation (str, optional): Activation function ('relu' or 'softmax').
            learning_rate (float, optional): Learning rate for Adam optimizer.
            beta1 (float, optional): Beta1 hyperparameter for Adam.
            beta2 (float, optional): Beta2 hyperparameter for Adam.
            epsilon (float, optional): Epsilon hyperparameter for Adam.
        """
        self.activation = activation
        # Initialize parameters
        self.w = np.random.randn(numNodesIn, numNodesOut) * np.sqrt(2. / numNodesIn)  # He Initialization
        self.b = np.zeros((1, numNodesOut))  # Zero Initialization
        self.gamma = np.ones((1, numNodesOut))  # Initialized to ones for scaling (BatchNorm)
        self.beta = np.zeros((1, numNodesOut))   # Initialized to zeros for shifting (BatchNorm)

        self.drop_out_prob = drop_out_prob
        self.dropout_mask = None
        self.prev_input = None
        self.z = None
        self.a = None
        self.delta = None
        self.n = None
        self.x_hat = None
        self.mean = None
        self.variance = None
        self.out = None
        self.cache = None
        self.training = True  # Default mode is training
        self.running_mean = np.zeros((1, numNodesOut))
        self.running_var = np.zeros((1, numNodesOut))

        # Initialize separate Adam Optimizers for each parameter
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
    
    # Forward Pass
    def forward(self, input):
        """
        Forward pass through the layer.

        Parameters:
            input (np.ndarray): Input data of shape (N, numNodesIn).

        Returns:
            np.ndarray: Activated output.
        """
        self.prev_input = input  # Shape: (N, numNodesIn)
        self.z = np.dot(input, self.w) + self.b  # Shape: (N, numNodesOut)
        
        if self.activation == 'relu':
            self.n = self.batchnorm_forward(self.z)  # Apply BatchNorm
            activated = self.relu(self.n)  # ReLU Activation
            self.a = self.dropout_forward(activated)  # Apply Dropout if applicable
        elif self.activation == 'softmax':
            self.a = self.softmax(self.z)  # Softmax Activation
        return self.a

    # Batch Normalization Forward Pass
    def batchnorm_forward(self, z):
        """
        Forward pass for Batch Normalization.

        Parameters:
            z (np.ndarray): Input to BatchNorm layer.

        Returns:
            np.ndarray: Batch-normalized output.
        """
        if self.training:
            mu = np.mean(z, axis=0, keepdims=True)
            var = np.var(z, axis=0, keepdims=True)
            self.x_hat = (z - mu) / np.sqrt(var + 1e-8)
            self.out = self.gamma * self.x_hat + self.beta
            
            # Update running mean and variance for inference
            momentum = 0.9
            self.running_mean = momentum * self.running_mean + (1 - momentum) * mu
            self.running_var = momentum * self.running_var + (1 - momentum) * var
            
            # Cache values for backward pass
            self.cache = (z, mu, var, self.x_hat)
        else:
            # Use running mean and variance during inference
            x_hat = (z - self.running_mean) / np.sqrt(self.running_var + 1e-8)
            self.out = self.gamma * x_hat + self.beta
        return self.out

    # Batch Normalization Backward Pass
    def batchnorm_backward(self, dout):
        """
        Backward pass for Batch Normalization.

        Parameters:
            dout (np.ndarray): Upstream gradients.

        Returns:
            tuple: Gradient w.r.t input z, gamma, and beta.
        """
        z, mu, var, x_hat = self.cache
        N, D = dout.shape

        # Gradients w.r.t gamma and beta
        dgamma = np.sum(dout * x_hat, axis=0, keepdims=True)
        dbeta = np.sum(dout, axis=0, keepdims=True)

        # Gradient w.r.t x_hat
        dxhat = dout * self.gamma

        # Gradient w.r.t variance
        dvar = np.sum(dxhat * (z - mu) * -0.5 * (var + 1e-8) ** (-1.5), axis=0, keepdims=True)

        # Gradient w.r.t mean
        dmu = np.sum(dxhat * -1 / np.sqrt(var + 1e-8), axis=0, keepdims=True) + \
              dvar * np.mean(-2 * (z - mu), axis=0, keepdims=True)

        # Gradient w.r.t z
        dz = dxhat / np.sqrt(var + 1e-8) + dvar * 2 * (z - mu) / N + dmu / N

        return dz, dgamma, dbeta

    # Dropout Forward Pass
    def dropout_forward(self, x):
        """
        Forward pass for Dropout.

        Parameters:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: Output after applying dropout.
        """
        if self.drop_out_prob is not None and self.training:
            self.dropout_mask = (np.random.rand(*x.shape) > self.drop_out_prob).astype(float)
            return x * self.dropout_mask / (1 - self.drop_out_prob)
        else:
            return x

    # Dropout Backward Pass
    def dropout_backward(self, dout):
        """
        Backward pass for Dropout.

        Parameters:
            dout (np.ndarray): Upstream gradients.

        Returns:
            np.ndarray: Gradient after applying dropout mask.
        """
        if self.drop_out_prob is not None and self.training:
            return dout * self.dropout_mask / (1 - self.drop_out_prob)
        else:
            return dout

    def backward(self, upstream_grad):
        """
        Backward pass through the layer.

        Parameters:
            upstream_grad (np.ndarray): Gradient from the upstream layer.

        Returns:
            np.ndarray: Gradient to pass to the previous layer.
        """
        if self.activation == 'relu':
            # Apply Dropout backward
            grad = self.dropout_backward(upstream_grad)
            
            # Apply ReLU derivative
            activation_derivative = self.relu_derivative(self.n)
            self.delta = grad * activation_derivative 
            
            if self.drop_out_prob is not None and self.training:
                self.delta *= self.dropout_mask 

            dz, dgamma, dbeta = self.batchnorm_backward(self.delta) 

            # Compute gradients w.r.t weights and biases
            gradient_w = np.dot(self.prev_input.T, dz) / self.prev_input.shape[0]  # Shape: (numNodesIn, numNodesOut)
            gradient_b = np.sum(dz, axis=0, keepdims=True) / self.prev_input.shape[0]  # Shape: (1, numNodesOut)

            # Update weights and biases using Adam Optimizer
            self.w_optimizer.update(self.w, gradient_w)
            self.b_optimizer.update(self.b, gradient_b)

            # Update gamma and beta using Adam Optimizer
            self.gamma_optimizer.update(self.gamma, dgamma / self.prev_input.shape[0])
            self.beta_optimizer.update(self.beta, dbeta / self.prev_input.shape[0])

            # Return gradient to the previous layer using updated weights
            return np.dot(dz, self.w.T)  # Shape: (N, numNodesIn)

        elif self.activation == 'softmax':
            self.delta = upstream_grad  # Shape: (N, D)

            # Gradients for weights and biases
            gradient_w = np.dot(self.prev_input.T, self.delta) / self.prev_input.shape[0]  # Shape: (numNodesIn, numNodesOut)
            gradient_b = np.sum(self.delta, axis=0, keepdims=True) / self.prev_input.shape[0]  # Shape: (1, numNodesOut)

            # Update weights and biases using Adam Optimizer
            self.w_optimizer.update(self.w, gradient_w)
            self.b_optimizer.update(self.b, gradient_b)

            # Return gradient to the previous layer
            return np.dot(self.delta, self.w.T)  # Shape: (N, numNodesIn)

    def set_training_mode(self, training=True):
        """
        Sets the layer to training or evaluation mode.

        Parameters:
            training (bool): True for training mode, False for evaluation mode.
        """
        self.training = training
    
    def save_parameters(self, file_path):
        """
        Saves the layer's parameters to a file.

        Parameters:
            file_path (str): Path to the file where parameters will be saved.
        """
        with open(file_path, 'wb') as f:
            pickle.dump({
                'w': self.w,
                'b': self.b,
                'gamma': self.gamma,
                'beta': self.beta
            }, f)

    def load_parameters(self, file_path):
        """
        Loads the layer's parameters from a file.

        Parameters:
            file_path (str): Path to the file from which parameters will be loaded.
        """
        with open(file_path, 'rb') as f:
            params = pickle.load(f)
            self.w = params['w']
            self.b = params['b']
            self.gamma = params['gamma']
            self.beta = params['beta']