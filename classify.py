import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from fnn_implement import forward_propagation, load_model

def predict_digits(image_path, model_path="mnist_model.noy"):
    
    # ------ Load trained parameters -----
    try:
        parameters = load_model(model_path)    
    except:
        print(f"File {model_path} not found.")
        return
    
    # ------  Image preprocessing ------
    
    