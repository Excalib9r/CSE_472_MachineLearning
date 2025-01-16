import numpy as np
import pickle
from Layers2 import Layers as ly 

class Network:
    def __init__(self, layers, drop_out = None, hidden_layer_activation='relu', output_layer_activation='softmax', loss_type='cross_entropy', learning_rate=0.0001, beta1 = 0.9, beta2 = 0.999):
        self.loss_type = loss_type
        self.layers = []  
        for i in range(len(layers) - 1):  
            if i == len(layers) - 2: 
                self.layers.append(ly(layers[i], layers[i + 1],  drop_out_prob=drop_out, activation=output_layer_activation, learning_rate=learning_rate, beta1=beta1, beta2=beta2))
            else:
                self.layers.append(ly(layers[i], layers[i + 1], activation=hidden_layer_activation,  learning_rate=learning_rate, beta1=beta1, beta2=beta2)) 
        self.output = None
    
    def forward(self, input, training=True):
        for layer in self.layers:
            if hasattr(layer, 'set_training_mode'):
                layer.set_training_mode(training)
            input = layer.forward(input)
        self.output = input
        return self.output
    
    def classify(self, input):
        self.output = self.forward(input, training=False)
        return np.argmax(self.output, axis=1)
    
    def calculateCost(self, target):
        output = self.output
        epsilon = 1e-12
        output = np.clip(output, epsilon, 1. - epsilon)
        return -np.sum(target * np.log(output)) / target.shape[0]
    
    def backward(self, target):
        loss = self.output - target
        for layer in reversed(self.layers):
            loss = layer.backward(loss)

    def save_parameters(self, file_path):
        parameters = {}
        for idx, layer in enumerate(self.layers):
            parameters[f'layer_{idx}'] = {
                'w': layer.w,
                'b': layer.b,
                'gamma': getattr(layer, 'gamma', None),
                'beta': getattr(layer, 'beta', None),
                'running_mean': layer.running_mean,
                'running_var': layer.running_var
            }
        with open(file_path, 'wb') as f:
            pickle.dump(parameters, f)

    def load_parameters(self, file_path):
        with open(file_path, 'rb') as f:
            parameters = pickle.load(f)
            for idx, layer in enumerate(self.layers):
                layer_params = parameters.get(f'layer_{idx}', {})
                if 'w' in layer_params:
                    layer.w = layer_params['w']
                if 'b' in layer_params:
                    layer.b = layer_params['b']
                if 'gamma' in layer_params and layer_params['gamma'] is not None:
                    layer.gamma = layer_params['gamma']
                if 'beta' in layer_params and layer_params['beta'] is not None:
                    layer.beta = layer_params['beta']
                if 'running_mean' in layer_params:
                    layer.running_mean = layer_params['running_mean']
                if 'running_var' in layer_params:
                    layer.running_var = layer_params['running_var']
                