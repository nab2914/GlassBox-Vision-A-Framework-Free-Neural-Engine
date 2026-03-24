import numpy as np
import matplotlib.pyplot as plt
from src.layers import DenseLayer
from src.activations import ReLU, Softmax
from src.losses import CategoricalCrossEntropy
from src.optimizers import SGD
from src.network import Sequential
import gradio as gr
import cv2
import os  
import csv

print("Initializing Deep Neural Network Engine...\n")

# # --- 1. LOAD MNIST DATA ---
# print("Loading MNIST Dataset (This may take a few seconds)...")
# # Using pure NumPy to load the CSV. 
# data = np.loadtxt('data/mnist.csv', delimiter=',', skiprows=1)
# # Check if your custom dataset exists, and if so, load it!
# if os.path.exists('data/my_handwriting.csv'):
#     print("Loading Custom Handwriting Data...")
#     custom_data = np.loadtxt('data/my_handwriting.csv', delimiter=',', skiprows=1)
    
#     # Glue your custom drawings to the bottom of the MNIST dataset
#     data = np.vstack((data, custom_data))
#     print(f"Added custom images to the training pool!")
# # The first column is the label (0-9), the rest are the 784 pixels

print("Loading ONLY Custom Handwriting Data...")
data = np.loadtxt('data/my_handwriting.csv', delimiter=',', skiprows=1)

X = data[:, 1:]
y_raw = data[:, 0].astype(int)

# Normalize pixel values from 0-255 down to 0.0-1.0
X = X / 255.0

# One-hot encode the labels (10 classes for digits 0-9)
num_samples = len(y_raw)
y = np.zeros((num_samples, 10))
y[np.arange(num_samples), y_raw] = 1

# Train/Test Split (80% Train, 20% Test)
split_idx = int(num_samples * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
print(f"Data Loaded: {X_train.shape[0]} Training Images, {X_test.shape[0]} Testing Images.\n")

# --- 2. BUILD THE DEEP ARCHITECTURE ---
model = Sequential()
# Input: 784 pixels -> Hidden 1: 128 neurons
model.add(DenseLayer(784, 128, init_method="he"))
model.add(ReLU())
# Hidden 1: 128 neurons -> Hidden 2: 64 neurons
model.add(DenseLayer(128, 64, init_method="he"))
model.add(ReLU())
# Hidden 2: 64 neurons -> Output: 10 classes (Digits 0-9)
model.add(DenseLayer(64, 10, init_method="xavier"))
model.add(Softmax())

loss_function = CategoricalCrossEntropy()
optimizer = SGD(learning_rate=0.1)

# --- 3. TRAINING LOOP WITH MINI-BATCHES ---
epochs = 50
batch_size = 16 # Process 128 images at a time
epoch_history, loss_history, accuracy_history = [], [], []

print("Starting Deep Learning Training...\n")
for epoch in range(epochs):
    
    # Shuffle the training data at the start of each epoch
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    epoch_loss = 0
    epoch_accuracy = 0
    batches = X_train.shape[0] // batch_size
    
    # Mini-Batch Loop
    for i in range(batches):
        # Slice the batch
        start = i * batch_size
        end = start + batch_size
        X_batch = X_train[start:end]
        y_batch = y_train[start:end]
        
        # Forward Pass
        predictions = model.forward(X_batch)
        
        # Calculate Loss & Accuracy for the batch
        batch_loss = loss_function.forward(predictions, y_batch)
        epoch_loss += batch_loss
        
        preds_idx = np.argmax(predictions, axis=1)
        true_idx = np.argmax(y_batch, axis=1)
        epoch_accuracy += np.mean(preds_idx == true_idx)
        
        # Backward Pass
        loss_grad = loss_function.backward()
        model.backward(loss_grad)
        
        # Optimization
        model.update_weights(optimizer)
    
    # Average metrics for the epoch
    avg_loss = epoch_loss / batches
    avg_acc = epoch_accuracy / batches
    
    epoch_history.append(epoch)
    loss_history.append(avg_loss)
    accuracy_history.append(avg_acc)
    
    print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {avg_acc * 100:.2f}%")

# --- 4. FINAL EVALUATION ---
print("\n--- Final Evaluation on Unseen Test Data ---")
test_predictions = model.forward(X_test)
test_loss = loss_function.forward(test_predictions, y_test)
test_acc = np.mean(np.argmax(test_predictions, axis=1) == np.argmax(y_test, axis=1))

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc * 100:.2f}%\n")

# Generate the new convergence plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epoch_history, loss_history, 'r-', linewidth=2)
plt.title('Deep Network Convergence (Loss)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epoch_history, accuracy_history, 'b-', linewidth=2)
plt.title('Deep Network Accuracy')
plt.grid(True)

plt.savefig('deep_convergence.png')
print("Saved 'deep_convergence.png'.")
print("\nLaunching Web Interface...")

def recognize_digit(drawing):
    if drawing is None or drawing["composite"] is None:
        return "Please draw a number!"
    
    image = drawing["composite"]
    
    # 1. Grayscale and Invert (White ink on Black background)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.bitwise_not(img_gray)
    
    # 2. Thresholding: Force all faint gray pixels to be solid white
    _, thresh = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)
    
    # 3. Find the exact bounding box of your drawing to crop out dead space
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return "Please draw a number!"
    x, y, w, h = cv2.boundingRect(coords)
    digit = thresh[y:y+h, x:x+w]
    
    # 4. Add perfect padding to make it a square (like the MNIST dataset)
    max_dim = max(w, h)
    pad_w = (max_dim - w) // 2
    pad_h = (max_dim - h) // 2
    # Add a 20-pixel border so the number doesn't touch the edges
    padded_digit = cv2.copyMakeBorder(digit, pad_h + 20, pad_h + 20, pad_w + 20, pad_w + 20, cv2.BORDER_CONSTANT, value=0)
    
    # 5. Shrink to exactly 28x28
    img_resized = cv2.resize(padded_digit, (28, 28), interpolation=cv2.INTER_AREA)
    
    # 6. Normalize and Flatten
    img_normalized = img_resized / 255.0
    img_flattened = img_normalized.reshape(1, 784)
    
    # 7. Predict!
    prediction = model.forward(img_flattened)
    predicted_number = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    
    return f"Prediction: {predicted_number} \nConfidence: {confidence:.2f}%"
    
def save_to_dataset(drawing, true_label):
    if drawing is None or drawing["composite"] is None:
        return "Please draw a number first."
    if not true_label.isdigit():
        return "Please type a valid number (0-9)."
        
    image = drawing["composite"]
    
    # Process the image exactly like we do for the AI
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.bitwise_not(img_gray)
    _, thresh = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    
    if coords is None:
        return "Canvas is blank!"
        
    x, y, w, h = cv2.boundingRect(coords)
    digit = thresh[y:y+h, x:x+w]
    
    max_dim = max(w, h)
    pad_w = (max_dim - w) // 2
    pad_h = (max_dim - h) // 2
    padded_digit = cv2.copyMakeBorder(digit, pad_h + 20, pad_h + 20, pad_w + 20, pad_w + 20, cv2.BORDER_CONSTANT, value=0)
    img_resized = cv2.resize(padded_digit, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Flatten into 784 pixels (keep them 0-255 for the CSV)
    pixels = img_resized.reshape(1, 784)[0].tolist()
    
    # Create the row: [Label, Pixel1, Pixel2... Pixel784]
    row = [int(true_label)] + pixels
    
    # Save to CSV
    file_path = 'data/my_handwriting.csv'
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            # Write header if it's a brand new file
            header = ["label"] + [f"pixel{i}" for i in range(1, 785)]
            writer.writerow(header)
        writer.writerow(row)
        
    return f"Success! Added your '{true_label}' to my_handwriting.csv"

# --- THE NEW GLASSBOX UI ---
with gr.Blocks(theme=gr.themes.Monochrome()) as interface:
    gr.Markdown("# GlassBox Engine: Vision Subsystem")
    gr.Markdown("Draw a digit(0-9). If the engine guesses wrong, correct it and save it to your personal dataset to make the AI smarter.")
    
    with gr.Row():
        with gr.Column():
            sketchpad = gr.Sketchpad(type="numpy", label="Draw Here",interactive=True,brush=gr.Brush(colors=["#000000"], default_size=20))
            predict_btn = gr.Button("Guess the Number", variant="primary")
            output_text = gr.Textbox(label="Engine Output")
            
        with gr.Column():
            gr.Markdown("### Continuous Learning (Data Flywheel)")
            correct_label = gr.Textbox(label="What number did you actually draw?")
            save_btn = gr.Button("Save to Personal Dataset")
            save_status = gr.Textbox(label="Database Status")
            
    # Connect the buttons to the functions
    predict_btn.click(fn=recognize_digit, inputs=sketchpad, outputs=output_text)
    save_btn.click(fn=save_to_dataset, inputs=[sketchpad, correct_label], outputs=save_status)

interface.launch()
