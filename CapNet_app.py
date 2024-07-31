

# This is the app.py file for CapNet Model


import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Layer, Conv2D
import os

# Define the Primary Capsule Layer
class PrimaryCapsuleLayer(Layer):
    def __init__(self, num_capsules, dim_capsule, kernel_size, strides, padding, **kwargs):
        super(PrimaryCapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.conv = Conv2D(filters=num_capsules * dim_capsule, kernel_size=kernel_size, strides=strides, padding=padding)

    def call(self, inputs):
        outputs = self.conv(inputs)
        outputs = tf.reshape(outputs, (-1, outputs.shape[1] * outputs.shape[2] * self.num_capsules, self.dim_capsule))
        return tf.keras.backend.sqrt(tf.reduce_sum(tf.square(outputs), -1))

# Register custom layer when loading the model
def load_capsnet_model(model_path):
    custom_objects = {'PrimaryCapsuleLayer': PrimaryCapsuleLayer}
    model = load_model(model_path, custom_objects=custom_objects)
    return model

# Load the model
model_path = 'CapsnetEfficientNet_Model_Brain_Tumor_Detection_Improved.h5'
model = load_capsnet_model(model_path)

# Set up the Streamlit app
st.title("Brain Tumor Detection using CapsNet Model")
st.write("Upload an MRI image to diagnose for the presence of a brain tumor.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type="jpg")

if uploaded_file is not None:
    # Load and preprocess the image
    img = image.load_img(uploaded_file, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions, axis=1)[0]

    # # Display results
    # class_labels = {0: 'Not Tumor', 1: 'Tumor'}
    # st.write(f"The model predicts that the image is: **{class_labels[predicted_class]}**")

    # Display the image
    st.image(img, caption='Uploaded MRI Image', use_column_width=True)

    if predicted_class == 0:
        st.write(f'Prediction: Not Tumor (Confidence: {confidence:.2f})')
    else:
        st.write(f'Prediction: Tumor (Confidence: {confidence:.2f})')

