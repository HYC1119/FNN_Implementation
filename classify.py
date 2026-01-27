import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from fnn_implement import forward_propagation, load_model

def predict_digits(image_path, model_path="mnist_model.npy"):
    
    # ------ Load trained parameters -----
    try:
        parameters = load_model(model_path)    
    except:
        print(f"File {model_path} not found.")
        return
    
    # ------ Image preprocessing ------
    # 1. convert into gray scale
    # 2. reverse color
    # 3. cut white frame and center the digit
    # 4. scale to 28*28 pixel with filter
    # 5. convert to array and normalize to 0~1
    # 6. flatten to 784 dimension

    img = Image.open(image_path).convert('L')
    img = ImageOps.invert(img)
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    
    img = img.resize((28,28), Image.Resampling.LANCZOS)
    img_array = np.array(img).astype("float32") / 255.0
    img_flatten = img_array.reshape(1, 784)

    # ------ Inference ------
    a3, _ = forward_propagation(img_flatten, parameters)
    prediction = np.argmax(a3, axis=1)[0]
    confidence = np.max(a3) * 100

    # ------ Show the result ------
    plt.imshow(img_array, cmap="gray")
    plt.title(f"Prediction: {prediction} {confidence: .2f}%")
    plt.axis("off")
    plt.show()
    
if __name__ == "__main__":
    image_name = "test_digit_3.PNG"
    weight_file = "mnist_model.npy"
    predict_digits(image_name, model_path=weight_file)

