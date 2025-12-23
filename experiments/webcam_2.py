import cv2
import numpy as np
import tensorflow as tf

# Load models
models = {
    "CKPLUS_MobileNetV2": tf.keras.models.load_model("ckplus_mobilenetv2.keras"),
    "RAFDB_MobileNetV2": tf.keras.models.load_model("mobilenetv2_rafdb_finetuned.keras"),
    "Emotion_CNN_Model_V2": tf.keras.models.load_model("emotion_cnn_model_v2_resumed.keras")
}

class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (224, 224))
        face_input = np.expand_dims(face_resized / 255.0, axis=0)

        model_predictions = []
        model_text = []

        # Run each model
        for model_name, m in models.items():
            preds = m.predict(face_input, verbose=0)[0]
            idx = np.argmax(preds)
            conf = float(preds[idx])

            model_predictions.append(preds)  
            model_text.append(f"{model_name}: {class_labels[idx]} {conf*100:.1f}%")

        # ENSEMBLE (AVERAGE PROBABILITIES)
        avg_probs = np.mean(model_predictions, axis=0)
        e_idx = np.argmax(avg_probs)
        e_label = class_labels[e_idx]
        e_conf = float(avg_probs[e_idx])

        # Draw box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Show ensemble output (big text)
        cv2.putText(frame, f"Ensemble: {e_label} {e_conf*100:.1f}%",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 0), 2)

        # Show per-model predictions below
        y_text = y + h + 20
        for t in model_text:
            cv2.putText(frame, t, (x, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            y_text += 25

    cv2.imshow("Real-Time Emotion Detection (3 Models + Ensemble)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
