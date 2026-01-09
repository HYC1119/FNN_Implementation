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

