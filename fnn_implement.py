import os 
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ====== Preprocessing ======
# Data loading
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "mnist.npz")

# Capture possible path error
try:
    data = np.load(data_path)
    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]
    data.close()
    print("✅ Loading Sucessful!")
except FileNotFoundError:
    print(f"Error: File not found, please ensure {data_path} exists.")
    exit()

# Flatten training data & testing data
x_train_flatten = x_train.reshape(x_train.shape[0], -1)
x_test_flatten = x_test.reshape(x_test.shape[0], -1)

# Normalize (avoid overflow)
x_train_norm = x_train_flatten.astype("float32") / 255.0
x_test_nome = x_test_flatten.astype("float32") / 255.0

# One-hot encoding 
num_class = 10
y_train_onehot = np.eye(num_class)[y_train]     # 10*10 unit matrix * y_train (number 0~9 matrix)
y_test_onehot = np.eye(num_class)[y_test]


# ====== Model Definition & Initialization ======
# Weight initialization
# Layer 1 (Input): 784 neurons
# Layer 2 (Hidden): 100 neurons 
# Layer 3 (Hidden): 150 neurons
# Layer 4 (Output): 10 neurons
input_neurons = 784
hidden1_neurons = 100
hidden2_neurons = 150 
output_neurons = 10

def initialize_parameters(input_neurons, hidden1_neurons, hidden2_neurons, output_neurons):
    np.random.seed(42)
    parameters = {    
        "W1": np.random.randn(784, 100) * 0.01,
        "b1": np.zeros((1, 100), dtype=float) * 0.01,
        "W2": np.random.randn(100, 150) * 0.01,
        "b2": np.zeros((1, 150), dtype=float) * 0.01,
        "W3": np.random.randn(150, 10) * 0.01,
        "b3": np.zeros((1, 10), dtype=float) * 0.01
    }
    return parameters

parameters = initialize_parameters(input_neurons, hidden1_neurons, hidden2_neurons, output_neurons)

# Define activation functions
def Relu(Z):                        # For forward propagation
    return np.maximum(0, Z)

def Relu_derivative(Z):             # For back propagation
    return (Z > 0).astype(float)

def softmax(Z):                     # For forward propagation
    # Avoide overflow
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    
    # Return propability distribution (sum = 1)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


# ====== Define forward propagation ======
def forward_propagation(X, parameters):
    # Extract weights and biases
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    W3, b3 = parameters["W3"], parameters["b3"]
    
    # Forward pass
    # First hidden layer
    Z1 = np.dot(X, W1) + b1
    A1 = Relu(Z1)
    
    # Second hidden layer
    Z2 = np.dot(A1, W2) + b2
    A2 = Relu(Z2)
    
    # Third hidden layer
    Z3 = np.dot(A2, W3) + b3
    A3 = Relu(Z3)    
    
    # Store values for backpropagation
    cache = {
        "Z1": Z1, "A1": A1, 
        "Z2": Z2, "A2": A2,
        "Z3": Z3, "A3": A3
    }
    
    return A3, cache

