import numpy as np
import matplotlib.pyplot as plt
from src.dataset import DatasetLoader
from src.layers import DenseLayer
from src.activations import ReLU, Softmax
from src.losses import CategoricalCrossEntropy
from src.optimizers import SGD

print("Initializing Framework-Free Neural Network...\n")

# 1. Load Data
loader = DatasetLoader('data/iris.csv')
loader.load_csv()
loader.normalize()
loader.one_hot_encode()
X_train, X_test, y_train, y_test = loader.train_test_split(test_ratio=0.2, random_seed=42)

# 2. Build Architecture
dense1 = DenseLayer(4, 8, init_method="he")
activation1 = ReLU()

dense2 = DenseLayer(8, 3, init_method="xavier")
activation2 = Softmax()

loss_function = CategoricalCrossEntropy()
optimizer = SGD(learning_rate=0.1)

# --- ANALYTICAL FRAMEWORK SETUP ---
epochs = 1000
epoch_history = []
loss_history = []
accuracy_history = []

print("\nStarting Training...\n")
for epoch in range(epochs):
    
    # Forward Pass
    Z1 = dense1.forward(X_train)
    A1 = activation1.forward(Z1)
    
    Z2 = dense2.forward(A1)
    A2 = activation2.forward(Z2)
    
    # Calculate Metrics
    loss = loss_function.forward(A2, y_train)
    predictions = np.argmax(A2, axis=1)
    true_labels = np.argmax(y_train, axis=1)
    accuracy = np.mean(predictions == true_labels)
    
    # Track Metrics for Plotting
    epoch_history.append(epoch)
    loss_history.append(loss)
    accuracy_history.append(accuracy)
    
    # Backward Pass
    loss_grad = loss_function.backward()
    activation2.backward(loss_grad)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    # Optimization
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {accuracy * 100:.2f}%")

print(f"\nFinal Epoch {epochs} | Loss: {loss:.4f} | Accuracy: {accuracy * 100:.2f}%")

# --- TESTING PHASE ---
Z1_test = dense1.forward(X_test)
A1_test = activation1.forward(Z1_test)
Z2_test = dense2.forward(A1_test)
A2_test = activation2.forward(Z2_test)

test_loss = loss_function.forward(A2_test, y_test)
test_accuracy = np.mean(np.argmax(A2_test, axis=1) == np.argmax(y_test, axis=1))

print("\n--- Model Evaluation on Unseen Data ---")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%\n")

# --- GENERATE PLOTS ---
print("Generating convergence plots...")
plt.figure(figsize=(12, 5))

# Plot 1: Loss over time
plt.subplot(1, 2, 1)
plt.plot(epoch_history, loss_history, label='Training Loss', color='red', linewidth=2)
plt.title('Neural Network Convergence (Loss)')
plt.xlabel('Epochs')
plt.ylabel('Categorical Cross-Entropy Loss')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Plot 2: Accuracy over time
plt.subplot(1, 2, 2)
plt.plot(epoch_history, accuracy_history, label='Training Accuracy', color='blue', linewidth=2)
plt.title('Training Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy Score')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.savefig('convergence_plot.png') # Saves the image to your folder
plt.show() # Opens the interactive window