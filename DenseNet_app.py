
# This is the app.py file For Dense Net Model


import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import os

# Load the trained model
model = None
model_path = 'DenseNet1_Model_brain_tumor_detection.h5'
st.write(f"Model path: {model_path}")

try:
    if os.path.exists(model_path):
        st.write("Model file found.")
        model = load_model(model_path)
        st.write("Model loaded successfully.")
    else:
        st.write(f"Model file '{model_path}' not found.")
except Exception as e:
    st.write("Error loading the model:", str(e))

# Define the class names
class_names = ['Not Tumor', 'Tumor']

# Define a function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Rescale image
    return image

# Streamlit app
def main():
    st.title("Brain Tumor Detection using DenseNet121")

    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load and preprocess the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Print preprocessed image shape for debugging
        st.write("Preprocessed Image Shape:", preprocessed_image.shape)  # Ensure (1, 224, 224, 3)

        # Make predictions
        try:
            if model is not None:
                predictions = model.predict(preprocessed_image)
                predicted_class = np.argmax(predictions, axis=1)[0]
                confidence = np.max(predictions)

                # Display the results
                st.write(f"Prediction: {class_names[predicted_class]}")
                st.write(f"Confidence: {confidence:.2f}")
            else:
                st.write("Model is not loaded. Please check the model file path.")
        except Exception as e:
            st.write("Error making predictions:", str(e))

# Entry point of the app
if __name__ == '__main__':
    main()
