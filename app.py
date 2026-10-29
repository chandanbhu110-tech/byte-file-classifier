import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("file_classifier.pkl")

st.title("Byte-Level File Type Classifier")
st.write("Upload a file to predict its type.")

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    data = uploaded_file.getvalue()

    if len(data) == 0:
        st.error("The uploaded file is empty.")
    else:
        # Extract byte-frequency features
        features = np.bincount(
            np.frombuffer(data, dtype=np.uint8),
            minlength=256
        ).astype(float)

        features = features / len(data)

        # Predict file type
        prediction = model.predict([features])[0]

        st.success(f"Predicted file type: {prediction.upper()}")
        st.write(f"Filename: {uploaded_file.name}")


 