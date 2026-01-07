import os 
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# load data
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "mnist.npz")

# capture possible path error
try:
    data = np.load(data_path)
    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]
    data.close()
    print("✅ Loading Sucessful!")
except FileNotFoundError:
    print(f"Error: File not found, please ensure {data_path} exists.")
    exit()
