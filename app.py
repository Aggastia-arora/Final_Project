import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model('object_detection.h5')

# Your label dictionary
labels_dictionary = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

st.title("Object Detection App ")
st.write("Upload an image and the model will predict the object!")
st.write("Upload only: Airplane, Car, Bird, Deer, Truck, Horse, Ship, Frog, Cat, Dog")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    image = image.resize((32, 32))
    image = np.array(image)
    image = image / 255.0  # Normalize if you trained with normalization
    image = np.expand_dims(image, axis=0)  # Model expects batch dimension

    # Make prediction
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)

    st.write(f"**Prediction:** {labels_dictionary[predicted_class]}")
