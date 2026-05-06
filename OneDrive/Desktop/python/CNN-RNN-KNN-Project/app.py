import streamlit as st
import numpy as np
from PIL import Image

from tensorflow.keras.models import load_model
import joblib

# -------------------------------
# Load Models
# -------------------------------

cnn_model = load_model("cnn_model.h5")
rnn_model = load_model("rnn_model.h5")
knn_model = joblib.load("knn_model.joblib")

# -------------------------------
# Streamlit Title
# -------------------------------

st.title("Digit Prediction App")

st.write("Upload a digit image and choose a model.")

# -------------------------------
# Model Selection
# -------------------------------

model_choice = st.selectbox(
    "Choose Model",
    ["CNN", "RNN", "KNN"]
)

# -------------------------------
# Upload Image
# -------------------------------

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg"]
)

# -------------------------------
# Prediction
# -------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")

    image = image.resize((28, 28))

    img_array = np.array(image)

    st.image(image, caption="Uploaded Image")

    img_array = img_array / 255.0

    # CNN
    if model_choice == "CNN":

        img = img_array.reshape(1, 28, 28, 1)

        prediction = cnn_model.predict(img)

    # RNN
    elif model_choice == "RNN":

        img = img_array.reshape(1, 28, 28)

        prediction = rnn_model.predict(img)

    # KNN
    else:

        img = img_array.reshape(1, 784)

        prediction = knn_model.predict(img)

        st.success(f"Predicted Digit: {prediction[0]}")

    # CNN/RNN Output
    if model_choice in ["CNN", "RNN"]:

        predicted_digit = np.argmax(prediction)

        st.success(f"Predicted Digit: {predicted_digit}")