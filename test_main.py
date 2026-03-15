import numpy as np
from src.dataset import DatasetLoader
from src.layers import DenseLayer
from src.activations import ReLU, Softmax
from src.losses import CategoricalCrossEntropy
from src.optimizers import SGD

print("Initializing Framework-Free Neural Network...\n")

# 1. Load and Prepare the Real Dataset
loader = DatasetLoader('data/iris.csv')
loader.load_csv()
loader.normalize()
loader.one_hot_encode()
X_train, X_test, y_train, y_test = loader.train_test_split(test_ratio=0.2, random_seed=42)

# 2. Build the Network Architecture
# Iris has 4 features, so input is 4. Let's use 8 neurons in the hidden layer. 
# There are 3 species, so the output layer must have 3 neurons.
dense1 = DenseLayer(4, 8, init_method="he")
activation1 = ReLU()

dense2 = DenseLayer(8, 3, init_method="xavier")
activation2 = Softmax()

loss_function = CategoricalCrossEntropy()

# 3. Configure the Optimizer (Learning Rate)
optimizer = SGD(learning_rate=0.1)

# 4. The Training Loop
epochs = 1000

print("\nStarting Training...\n")
for epoch in range(epochs):
    
    # --- FORWARD PASS ---
    # Layer 1
    dense1.forward(X_train)
    activation1.forward(dense1.inputs)
    
    Z1 = dense1.forward(X_train)
    A1 = activation1.forward(Z1)
    
    # Layer 2
    Z2 = dense2.forward(A1)
    A2 = activation2.forward(Z2)
    
    # Calculate Loss
    loss = loss_function.forward(A2, y_train)
    
    # Calculate Accuracy 
    # (Compare the index of the highest probability to the index of the true label)
    predictions = np.argmax(A2, axis=1)
    true_labels = np.argmax(y_train, axis=1)
    accuracy = np.mean(predictions == true_labels)
    
    # --- BACKWARD PASS ---
    # Gradient of the loss
    loss_grad = loss_function.backward()
    
    # Backpropagate through Layer 2
    activation2.backward(loss_grad)
    dense2.backward(activation2.dinputs)
    
    # Backpropagate through Layer 1
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    # --- OPTIMIZATION ---
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    
    # Print metrics every 100 epochs to monitor the "thermometer"
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {accuracy * 100:.2f}%")

print(f"\nFinal Epoch {epochs} | Loss: {loss:.4f} | Accuracy: {accuracy * 100:.2f}%")
print("\nTraining Complete!")

# --- TESTING PHASE ---
print("\n--- Model Evaluation on Unseen Data ---")

# 1. Forward pass through Layer 1 using the test dataset
Z1_test = dense1.forward(X_test)
A1_test = activation1.forward(Z1_test)

# 2. Forward pass through Layer 2
Z2_test = dense2.forward(A1_test)
A2_test = activation2.forward(Z2_test)

# 3. Calculate Loss on Test Data
test_loss = loss_function.forward(A2_test, y_test)

# 4. Calculate Accuracy on Test Data
test_predictions = np.argmax(A2_test, axis=1)
true_test_labels = np.argmax(y_test, axis=1)
test_accuracy = np.mean(test_predictions == true_test_labels)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%\n")