import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tensorflow.keras.models import load_model
import pickle

warnings.filterwarnings("ignore")

def preprocessing(image,model):
    img = np.array(image,dtype=np.uint8)  
    
    # Ensure it's a NumPy array

    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Apply Adaptive Thresholding
    img_thresh = cv2.adaptiveThreshold(
        img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 51, 0  # Adjust parameters as needed
    )

    # --- Resize while keeping aspect ratio and pad to 28x28 ---
    h, w = img_thresh.shape
    aspect_ratio = w / h

    if aspect_ratio > 1:
        new_w = 28
        new_h = int(28 / aspect_ratio)
    else:
        new_h = 28
        new_w = int(28 * aspect_ratio)

    # Resize the image to the new dimensions
    img_resized = cv2.resize(img_thresh, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a blank 28x28 white canvas
    img_padded = np.ones((28, 28), dtype=np.uint8) * 255

    # Compute offsets to center the resized image
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    # Place the resized image on the white canvas
    img_padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized

    # Normalize pixel values to [0, 1]
    processed_image = img_padded.astype('float32') / 255.0

    # Display the final processed image
    plt.figure(figsize=(4, 4))
    plt.imshow(processed_image, cmap='gray')
    plt.title("Final Processed Image Before Prediction")
    plt.axis('off')
    plt.show()

    # Reshape for model input
    model_input = processed_image.reshape(1, 28, 28, 1)
    return model_input
    

def main():
    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct absolute path to the test image
    image_path = os.path.join(current_dir, 'test', '2.png')
    
    # Check if file exists before reading
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
        
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image at {image_path}")
        return
    model = load_model("model.h5")    
    preprocessing(image,model)
    
if __name__ == "__main__":
    main()
