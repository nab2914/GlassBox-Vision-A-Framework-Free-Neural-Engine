# 🔍 GlassBox Vision: A Framework-Free Neural Engine

Most modern Artificial Intelligence operates as a "black box" using heavy libraries like PyTorch or TensorFlow. **GlassBox Vision** was engineered to open that box. 

This repository contains a Deep Neural Network built entirely from first principles using pure Python and NumPy matrix algebra. It successfully classifies the MNIST Computer Vision dataset and features a real-time web interface with a continuous learning pipeline. 

## ✨ Key Features

* **Zero-Framework Architecture:** No Keras, PyTorch, or TensorFlow. Every layer, activation function, and optimization step is mathematically calculated from scratch.
* **Deep Computer Vision:** A multi-layer architecture (784 → 128 → 64 → 10) utilizing `ReLU` activations to solve the vanishing gradient problem, achieving 96%+ accuracy on unseen MNIST data.
* **Custom Calculus Engine:** Features a hand-coded Backpropagation engine using the Chain Rule, Categorical Cross-Entropy Loss, and Mini-Batch Stochastic Gradient Descent (SGD).
* **Interactive UI:** Deployed locally via `Gradio` and `OpenCV`. Users can draw digits on a digital sketchpad, which the engine normalizes, centers, and predicts in real-time.
* **Continuous Learning (Data Flywheel):** A built-in feedback loop. If the engine misclassifies a digit, the user can correct it. The pipeline automatically extracts the 784 pixels, formats them, and saves them to a custom `my_handwriting.csv` dataset to actively retrain and fine-tune the network's weights.

## 🧠 The Math Inside the Box
Instead of calling `model.fit()`, this engine manually executes:
1. **Forward Pass:** $Z = XW + b$ 
2. **Activations:** Custom `ReLU` for hidden layers, `Softmax` for probability distribution.
3. **Loss Calculation:** Measuring the exact mathematical error.
4. **Backward Pass:** Pushing the error gradients in reverse to calculate exact weight contributions.
5. **Optimization:** Physically adjusting the NumPy weight matrices to step down the error curve.

## 🚀 Getting Started

**1. Install Dependencies**
```bash
pip install numpy matplotlib gradio opencv-python