import cv2
import tensorflow as tf
import numpy as np

model_paths = {
    "CKPLUS_MobileNetV2"       : "ckplus_mobilenetv2.keras",
    "RAFDB_MobileNetV2"        : "mobilenetv2_rafdb_finetuned.keras",
    "Emotion_CNN_Model_V2"     : "emotion_cnn_model_v2_resumed.keras"
}

models = {}

for name, path in model_paths.items():
    print(f"Loading: {name}")
    models[name] = tf.keras.models.load_model(path)


class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Haar face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
print("\nReal-time emotion comparison started...")
print("Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        face_img = frame[y:y+h, x:x+w]

        predictions = {}

        for name, model in models.items():

            # Extract model input details
            _, H, W, C = model.input_shape   # C = 1 or 3

            # Resize face
            resized = cv2.resize(face_img, (W, H))

            # If model requires GRAYSCALE
            if C == 1:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                resized = np.expand_dims(resized, axis=-1)

            resized = resized.astype("float32") / 255.0
            resized = np.expand_dims(resized, axis=0)

            preds = model.predict(resized, verbose=0)
            label_index = np.argmax(preds[0])
            confidence = np.max(preds[0])

            predictions[name] = f"{class_labels[label_index]} ({confidence:.2f})"

        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display predictions (stacked above face)
        offset = 20
        for i, (name, pred) in enumerate(predictions.items()):
            cv2.putText(
                frame,
                f"{name}: {pred}",
                (x, y - 10 - i*offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

    cv2.imshow("Real-Time Emotion – Multi-Model", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
