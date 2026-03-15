import numpy as np

class DenseLayer:
    def __init__(self, input_size, output_size, init_method="random"):
        """Initializes the weights and biases for the dense layer."""
        self.input_size = input_size
        self.output_size = output_size
        self.init_method = init_method.lower()
        
        if self.init_method == "he":
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        elif self.init_method == "xavier":
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)
        else:
            self.weights = np.random.randn(input_size, output_size) * 0.01
            
        self.biases = np.zeros((1, output_size))
        
        # Variables to store data for the backward pass
        self.inputs = None
        self.dweights = None
        self.dbiases = None
        self.dinputs = None

    def forward(self, inputs):
        """Performs the forward pass: Z = XW + b"""
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        """
        Performs the backward pass to calculate gradients.
        dvalues: The gradient of the loss with respect to the output of this layer.
        """
        # Gradients on parameters (Weights and Biases)
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # Gradient on values (to pass back to the previous layer)
        self.dinputs = np.dot(dvalues, self.weights.T)