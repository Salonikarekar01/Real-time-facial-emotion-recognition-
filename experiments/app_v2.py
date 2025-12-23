import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import time
import pandas as pd
from PIL import Image
import io

MODEL_PATHS = {
    "CKPLUS_MobileNetV2": "ckplus_mobilenetv2.keras",
    "RAFDB_MobileNetV2": "mobilenetv2_rafdb_finetuned.keras",
    "Emotion_CNN_Model_V2": "mobilenetv2_finetuned.keras"
}

@st.cache_resource
def load_models():
    models = {}
    for name, path in MODEL_PATHS.items():
        try:
            models[name] = tf.keras.models.load_model(path)
        except:
            st.error(f"Failed to load: {name}")
    return models

models = load_models()

class_labels = ['angry','disgust','fear','happy','neutral','sad','surprise']


# UTILITIES
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                     "haarcascade_frontalface_default.xml")


def preprocess_for_model(image_bgr, model):
    _, H, W, C = model.input_shape

    img = cv2.resize(image_bgr, (W, H))

    if C == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=-1)

    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    return img


def predict_all_models(face_img):
    results = {}
    for model_name, model in models.items():

        input_tensor = preprocess_for_model(face_img, model)

        start = time.time()
        pred = model.predict(input_tensor, verbose=0)
        end = time.time()

        label_idx = np.argmax(pred[0])
        confidence = np.max(pred[0])

        results[model_name] = {
            "label": class_labels[label_idx],
            "confidence": float(confidence),
            "time_ms": round((end - start) * 1000, 2)
        }

    return results


def generate_report(results):
    df = pd.DataFrame(results).T
    buffer = io.StringIO()
    df.to_csv(buffer)
    return buffer.getvalue()


# STREAMLIT UI DESIGN
st.set_page_config(
    page_title="Emotion Recognition Dashboard",
    layout="wide",
    page_icon="🎭",
)

st.markdown("""
<style>
body { background-color: #f5f9ff; }
</style>
""", unsafe_allow_html=True)

st.title("🎭 Emotion Recognition – Multi-Model Dashboard")
st.write("Compare the performance of **CKPLUS**, **RAF-DB**, and **CNN_V2** on webcam or uploaded images.")

input_mode = st.sidebar.radio(
    "Input Mode",
    ["Upload Image", "Use Webcam"],
    index=0
)

face_detect_toggle = st.sidebar.checkbox("Enable Face Detection", value=True)


# MODE 1 — IMAGE UPLOAD
if input_mode == "Upload Image":

    uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

    if uploaded:
        img = Image.open(uploaded)
        img_rgb = np.array(img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        col1, col2 = st.columns([1,1])
        col1.image(img_rgb, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze Image"):
            display = img_bgr.copy()

            faces = [(0,0,img_bgr.shape[1], img_bgr.shape[0])] if not face_detect_toggle \
                else face_detector.detectMultiScale(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),1.3,5)

            all_results = {}

            for i, (x,y,w,h) in enumerate(faces):
                face_crop = img_bgr[y:y+h, x:x+w]
  #               # PREPROCESS face for NN inputs
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_resized = cv2.resize(face_rgb, (224, 224))
                face_norm = face_resized.astype("float32") / 255.0
                face_input = np.expand_dims(face_norm, axis=0)

                # RUN ALL MODELS
                all_results = {}
                for model_name, model in loaded_models.items():
                    preds = model.predict(face_input, verbose=0)[0]
                    label = class_labels[np.argmax(preds)]
                    conf = float(np.max(preds))
                    all_results[model_name] = (label, conf)

                # DISPLAY PREDICTIONS
                st.subheader("Prediction Results for This Face")
                for model_name, (label, conf) in all_results.items():
                    st.metric(
                        label=f"{model_name}",
                        value=f"{label} ({conf*100:.2f}%)"
                    )
#
                results = predict_all_models(face_crop)
                all_results[f"Face {i+1}"] = results

                cv2.rectangle(display,(x,y),(x+w,y+h),(0,255,0),2)

            col2.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), caption="Detected Faces", use_column_width=True)

            st.subheader("📊 Model Predictions per Face")
            for face, res in all_results.items():
                st.markdown(f"### {face}")
                df = pd.DataFrame(res).T
                st.dataframe(df)

                st.bar_chart(df["confidence"])

            csv = generate_report(all_results)
            st.download_button("⬇ Download Predictions Report", csv, "emotion_report.csv")


# MODE 2 — WEBCAM LIVE DEMO
else:
    st.write("Press **Start** to begin webcam.")

    start_cam = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])

    cap = None

    if start_cam:
        cap = cv2.VideoCapture(0)

    while start_cam:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        faces = [(0,0,frame.shape[1],frame.shape[0])] if not face_detect_toggle else \
            face_detector.detectMultiScale(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),1.3,5)

        y_text = 20
        for (x,y,w,h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            results = predict_all_models(face_crop)

            # Draw prediction on frame
            for model_name, r in results.items():
                text = f"{model_name}: {r['label']} ({r['confidence']:.2f})"
                cv2.putText(display, text, (10, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
                y_text += 25

            cv2.rectangle(display,(x,y),(x+w,y+h),(255,0,0),2)

        FRAME_WINDOW.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))

    if cap:
        cap.release()
