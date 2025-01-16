import numpy as np
from Layers2 import Layers as ly 

class Network:
    def __init__(self, layers, learning_rate = 0.001 , hidden_layer_activation='relu', output_layer_activation='softmax', loss_type='cross_entropy'):
        self.loss_type = loss_type
        self.learning_rate = learning_rate
        self.layers = []  # hidden layers including input and output layers
        for i in range(len(layers) - 1):  # index 0 to n-2
            if i == len(layers) - 2:  # Check if it's the last layer
                self.layers.append(ly(layers[i], layers[i + 1], activation=output_layer_activation))
            else:
                self.layers.append(ly(layers[i], layers[i + 1], activation=hidden_layer_activation)) 
        self.output = None
    
    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        self.output = input
        return self.output

    def classify(self, input):
        self.output = self.forward(input)
        return np.argmax(self.output, axis=1)
    
    def calculateCost(self, target):
        output = self.output
        return -np.sum(target * np.log(output + 1e-9)) / target.shape[0]
    
    def backward(self, target):
        loss = self.output - target
        for layer in reversed(self.layers):
            loss = layer.backward(loss, self.learning_rate)
        