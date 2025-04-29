import streamlit as st
import numpy as np
import tensorflow as tf
import os
from PIL import Image
import sys
import cv2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Extend system path to import custom module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.image_preprocessing import preprocessing

from streamlit_drawable_canvas import st_canvas

def load_model():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        model_path = os.path.join(parent_dir, "model", "model.h5")

        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            return None
        
        if os.path.getsize(model_path) == 0:
            st.error("Model file is empty.")
            return None

        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        return model

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Handwritten Digit Classifier(0-9)",
        page_icon="✍️",
        layout="wide"
    )

    st.title("Handwritten Digit Classification(0-9)")

    col1, col2 = st.columns([4, 1])

    model = load_model()
    if model is None:
        return

    with col1:
        st.subheader("Draw a digit below 👇")
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=15,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
            display_toolbar=st.sidebar.checkbox("Display toolbar", True),
        )

    with col2:
        st.subheader("Prediction")

        if canvas_result.image_data is not None and np.any(canvas_result.image_data != 0):
            if st.button("Predict"):
                try:
                    # Convert RGBA canvas to grayscale PIL image
                    img = Image.fromarray((canvas_result.image_data).astype('uint8'), mode="RGBA")
                    img = img.convert('L')  # Convert to grayscale

                    # Convert grayscale image to 3-channel BGR format
                    img_np_gray = np.array(img)
                    img_bgr = cv2.cvtColor(img_np_gray, cv2.COLOR_GRAY2BGR)

                    # Preprocess and predict
                    predicted_class = preprocessing(img_bgr,model)  # Your preprocessing file handles prediction
                    
                    if predicted_class is not None:
                         # Predict the digit using the trained model
                        predictions = model.predict(predicted_class)
                        predicted_class = np.argmax(predictions, axis=1)[0]
                        st.write(f"Predicted Digit: {predicted_class}")
                    else:
                        st.write("It is none")

                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
        else:
            st.info("Please draw a digit to predict.")

if __name__ == "__main__":
    main()
