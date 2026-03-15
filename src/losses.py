import numpy as np

class MeanSquaredError:
    def forward(self, y_pred, y_true):
        """
        Calculates the Mean Squared Error.
        Formula: L = (1/n) * sum((y_true - y_pred)^2)
        """
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean(np.power(self.y_true - self.y_pred, 2))

    def backward(self):
        """
        Calculates the derivative of the MSE with respect to the predictions.
        Formula: dL/dy_pred = (2/n) * (y_pred - y_true)
        """
        samples = self.y_pred.shape[0]
        # We return the gradient scaled by the number of samples
        return (2 / samples) * (self.y_pred - self.y_true)


class CategoricalCrossEntropy:
    def forward(self, y_pred, y_true):
        """
        Calculates the Categorical Cross-Entropy loss.
        """
        self.y_pred = y_pred
        self.y_true = y_true
        
        # Clip data to prevent division by 0 or log(0)
        # Clip predictions to range [1e-7, 1 - 1e-7]
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # Calculate cross-entropy: -sum(y_true * log(y_pred))
        # We use np.sum on axis=1 to sum across the classes for each sample
        sample_losses = -np.sum(y_true * np.log(y_pred_clipped), axis=1)
        
        # Return the average loss across the batch
        return np.mean(sample_losses)

    def backward(self):
        """
        Calculates the derivative of Categorical Cross-Entropy.
        """
        samples = self.y_pred.shape[0]
        y_pred_clipped = np.clip(self.y_pred, 1e-7, 1 - 1e-7)
        
        # Derivative formula: - (y_true / y_pred) / samples
        return -(self.y_true / y_pred_clipped) / samples