import numpy as np

class DenseLayer:
    def __init__(self, numNodesIn, numNodesOut):
        # He Initialization for weights if using ReLU
        self.w = np.random.randn(numNodesIn, numNodesOut) * np.sqrt(2. / numNodesIn)
        self.b = np.zeros((1, numNodesOut))  # Zero Initialization
        self.prev_input = None
        self.gradient_w = None
        self.gradient_b = None

    def forward(self, input):
        self.prev_input = input  # Shape: (N, numNodesIn)
        self.z = np.dot(input, self.w) + self.b  # Shape: (N, numNodesOut)
        return self.z

    def backward(self, upstream_grad, learning_rate, optimizer=None):
        # Compute gradients
        self.gradient_w = np.dot(self.prev_input.T, upstream_grad) / self.prev_input.shape[0]
        self.gradient_b = np.sum(upstream_grad, axis=0, keepdims=True) / self.prev_input.shape[0]
        
        # Update parameters
        if optimizer:
            self.w, self.b = optimizer.update(self.w, self.gradient_w, self.b, self.gradient_b)
        else:
            self.w -= learning_rate * self.gradient_w
            self.b -= learning_rate * self.gradient_b

        # Return gradient to previous layer
        return np.dot(upstream_grad, self.w.T)  # Shape: (N, numNodesIn)

class BatchNormLayer:
    def __init__(self, numNodesOut, momentum=0.9, epsilon=1e-8):
        self.gamma = np.ones((1, numNodesOut))  # Scale parameter
        self.beta = np.zeros((1, numNodesOut))  # Shift parameter
        self.momentum = momentum
        self.epsilon = epsilon
        self.cache = None
        self.training = True
        self.running_mean = np.zeros((1, numNodesOut))
        self.running_var = np.zeros((1, numNodesOut))
        self.gradient_gamma = None
        self.gradient_beta = None

    def forward(self, z):
        if self.training:
            mu = np.mean(z, axis=0, keepdims=True)
            xmu = z - mu
            sq = xmu ** 2
            var = np.mean(sq, axis=0, keepdims=True)
            sqrtvar = np.sqrt(var + self.epsilon)
            ivar = 1. / sqrtvar
            xhat = xmu * ivar
            gammax = self.gamma * xhat
            out = gammax + self.beta

            # Update running averages
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

            # Store cache for backward pass
            self.cache = (xhat, xmu, ivar, sqrtvar, var)
        else:
            xhat = (z - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            gammax = self.gamma * xhat
            out = gammax + self.beta

        return out

    def backward(self, dout, learning_rate, optimizer=None):
        xhat, xmu, ivar, sqrtvar, var = self.cache
        N, D = dout.shape

        # Gradients w.r.t. gamma and beta
        self.gradient_gamma = np.sum(dout * xhat, axis=0, keepdims=True)
        self.gradient_beta = np.sum(dout, axis=0, keepdims=True)

        # Gradients w.r.t. input
        dxhat = dout * self.gamma
        divar = np.sum(dxhat * xmu, axis=0, keepdims=True)
        dxmu1 = dxhat * ivar
        dsqrtvar = -1. / (sqrtvar**2) * divar
        dvar = 0.5 * 1. / np.sqrt(var + self.epsilon) * dsqrtvar
        dsq = (1. / N) * np.ones((N, D)) * dvar
        dxmu2 = 2 * xmu * dsq
        dx1 = dxmu1 + dxmu2
        dmu = -1 * np.sum(dxmu1 + dxmu2, axis=0, keepdims=True)
        dx2 = (1. / N) * np.ones((N, D)) * dmu
        dx = dx1 + dx2

        # Update gamma and beta
        if optimizer:
            self.gamma, self.beta = optimizer.update(self.gamma, self.gradient_gamma, self.beta, self.gradient_beta)
        else:
            self.gamma -= learning_rate * self.gradient_gamma
            self.beta -= learning_rate * self.gradient_beta

        return dx

class DropoutLayer:
    def __init__(self, drop_out_prob=None):
        self.drop_out_prob = drop_out_prob
        self.dropout_mask = None
        self.training = True

    def forward(self, x):
        if self.drop_out_prob is not None and self.training:
            self.dropout_mask = (np.random.rand(*x.shape) > self.drop_out_prob).astype(float)
            return x * self.dropout_mask / (1 - self.drop_out_prob)
        else:
            return x

    def backward(self, dout):
        if self.drop_out_prob is not None and self.training:
            return dout * self.dropout_mask / (1 - self.drop_out_prob)
        else:
            return dout

class MetaLayer:
    def __init__(self, numNodesIn, numNodesOut, drop_out_prob=None, activation='relu'):
        self.dense = DenseLayer(numNodesIn, numNodesOut)
        self.batchnorm = BatchNormLayer(numNodesOut)
        self.dropout = DropoutLayer(drop_out_prob) if activation == 'relu' else None
        self.activation = activation

    def forward(self, input):
        out = self.dense.forward(input)
        out = self.batchnorm.forward(out)
        if self.activation == 'relu':
            out = np.maximum(0, out)  # ReLU Activation
            if self.dropout:
                out = self.dropout.forward(out)
        elif self.activation == 'softmax':
            exps = np.exp(out - np.max(out, axis=1, keepdims=True))
            out = exps / np.sum(exps, axis=1, keepdims=True)
        return out

    def backward(self, upstream_grad, learning_rate, optimizer=None):
        if self.activation == 'relu':
            if self.dropout:
                upstream_grad = self.dropout.backward(upstream_grad)
            upstream_grad = upstream_grad * (self.dense.z > 0)
        elif self.activation == 'softmax':
            # Assuming loss gradient is already computed
            pass

        # Backward pass through BatchNorm
        upstream_grad = self.batchnorm.backward(upstream_grad, learning_rate, optimizer)

        # Backward pass through Dense layer
        upstream_grad = self.dense.backward(upstream_grad, learning_rate, optimizer)

        return upstream_grad