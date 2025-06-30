import streamlit as st
import numpy as np
import pickle
import face_recognition
import cv2

# ------------------------------
# Load your trained model & scaler
# ------------------------------
with open('trained_model.pkl', 'rb') as f:
    trained_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# ------------------------------
# Extract features (from above!)
# ------------------------------
def extract_face_features(image_array):
    rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    if len(face_locations) == 0:
        raise ValueError("No face found!")
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    if len(face_encodings) == 0:
        raise ValueError("No encodings found!")

    features = np.array(face_encodings[0])
    if features.shape[0] < 256:
        features = np.pad(features, (0, 256 - features.shape[0]))
    elif features.shape[0] > 256:
        features = features[:256]
    return features

# ------------------------------
# Predict identity
# ------------------------------
def predict_face_identity(face_features):
    scaled = scaler.transform([face_features])
    label = trained_model.predict(scaled)[0]
    prob = trained_model.predict_proba(scaled).max()
    return label, prob

# ------------------------------
# Streamlit app UI
# ------------------------------
st.set_page_config(page_title="Face Recognition System", page_icon=":bust_in_silhouette:")

st.title("🔍 Face Recognition System (dlib)")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)

    if st.button("Recognize Face"):
        try:
            features = extract_face_features(image)
            label, prob = predict_face_identity(features)
            st.success(f"**Predicted Identity:** {label}")
            st.info(f"**Confidence:** {prob*100:.2f}%")
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---\n✅ *Built with Streamlit + face_recognition*")
