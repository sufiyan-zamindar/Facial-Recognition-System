import streamlit as st
import cv2
import numpy as np
import joblib

# -------------------------------------------------
# ✅ Set page config FIRST — before anything runs
# -------------------------------------------------
st.set_page_config(page_title="Face Verification (No dlib)", page_icon=":bust_in_silhouette:")

# -------------------------------------------------
# Load Haar Cascade for face detection
# -------------------------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# -------------------------------------------------
# Load your trained ML model
# -------------------------------------------------
@st.cache_resource
def load_my_model():
    model = joblib.load('best_trained_model.pkl')  # ❌ This only runs if you call load_my_model()
    return model

model = load_my_model()


# -------------------------------------------------
# Extract simple pixel features from the face
# -------------------------------------------------
def extract_simple_features(face_img):
    resized = cv2.resize(face_img, (15, 10))  # 32x32 = 1024 features
    features = resized.flatten() / 255.0  # Normalize pixel values
    return features.reshape(1, -1)

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.title("🔍 Face Verification System ")

st.write(
    "✅ Upload an image. The system will:\n"
    "- Detect the face using Haar Cascade\n"
    "- Extract simple pixel features\n"
    "- Use ML model to verify if the person is authorized"
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        st.error("No face detected!")
    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_img = gray[y:y + h, x:x + w]
            features = extract_simple_features(face_img)
            prediction = model.predict(features)[0]
            proba = model.predict_proba(features)[0].max()

            if prediction == 1:
                st.success(f"✅ Person Verified! (Confidence: {proba:.2f})")
            else:
                st.error(f"❌ Person Not Verified (Confidence: {proba:.2f})")

        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                 caption="Detected Face(s)",
                 use_container_width=True)

st.markdown("---\n✅ *Built with Streamlit + OpenCV Haar Cascade — No dlib required!*")
