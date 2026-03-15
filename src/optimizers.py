class SGD:
    def __init__(self, learning_rate=1.0):
        """
        Initializes the Stochastic Gradient Descent optimizer.
        
        Args:
            learning_rate (float): The step size for weight updates.
        """
        self.learning_rate = learning_rate

    def update_params(self, layer):
        """
        Updates the weights and biases of a layer using the calculated gradients.
        """
        # We subtract the gradient multiplied by the learning rate
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases