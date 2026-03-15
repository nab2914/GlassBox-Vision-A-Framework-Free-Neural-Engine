class Sequential:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        # Pass data through all layers sequentially
        layer_output = X
        for layer in self.layers:
            layer_output = layer.forward(layer_output)
        return layer_output

    def backward(self, loss_gradient):
        # Pass error backward through all layers
        gradient = loss_gradient
        for layer in reversed(self.layers):
            layer.backward(gradient)
            gradient = layer.dinputs

    def update_weights(self, optimizer):
        # Update weights for any layer that has them
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                optimizer.update_params(layer)