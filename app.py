import streamlit as st
import numpy as np
import pickle
from deepface import DeepFace
import cv2
from sklearn.preprocessing import MinMaxScaler

# ------------------------------
# Load your saved model & scaler
# ------------------------------
# Example paths - replace with yours
with open('trained_model.pkl', 'rb') as f:
    trained_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# ------------------------------
# Feature Extractor with FaceNet
# ------------------------------
def extract_face_features(image_array):
    """
    Takes a raw face image (numpy array), saves it temporarily,
    then uses DeepFace FaceNet to extract embedding.
    Returns a 256-d vector.
    """
    temp_image_path = "temp_uploaded_image.jpg"
    cv2.imwrite(temp_image_path, image_array)

    # Get embedding using DeepFace + FaceNet
    embedding_obj = DeepFace.represent(
        img_path=temp_image_path,
        model_name='Facenet',
        enforce_detection=True
    )

    embedding = embedding_obj[0]["embedding"]
    feature_vector = np.array(embedding)

    # Truncate or pad to match 256 dims
    if feature_vector.shape[0] > 256:
        feature_vector = feature_vector[:256]
    elif feature_vector.shape[0] < 256:
        feature_vector = np.pad(feature_vector, (0, 256 - feature_vector.shape[0]))

    return feature_vector

# ------------------------------
# Predict identity
# ------------------------------
def predict_face_identity(face_features):
    scaled_features = scaler.transform([face_features])
    predicted_label = trained_model.predict(scaled_features)[0]
    probability = trained_model.predict_proba(scaled_features).max()
    return predicted_label, probability

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(
    page_title="Face Recognition System",
    page_icon=":bust_in_silhouette:",
    layout="centered"
)

st.title("🔍 Face Recognition System")
st.markdown(
    """
    Upload a face image to identify the person using FaceNet embeddings.
    """
)

# File uploader
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)

    if st.button("Recognize Face"):
        try:
            # Extract features
            face_features = extract_face_features(image)

            # Predict identity
            predicted_label, probability = predict_face_identity(face_features)

            st.success(f"**Predicted Identity:** {predicted_label}")
            st.info(f"**Confidence:** {probability*100:.2f}%")

        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---\n🚀 *Built with Streamlit & DeepFace*")
