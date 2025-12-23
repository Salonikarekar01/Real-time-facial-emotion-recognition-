# evaluate_cnn.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

test_dir = "/Users/salonikarekar/Desktop/College work/sem 5/sentiment analysis project files /image_sentiment_FER_Dataset/test"

model = tf.keras.models.load_model("emotion_cnn_model_v2_resumed.keras")

# Image preprocessing (grayscale)
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    color_mode='grayscale',        # match training
    batch_size=32,
    class_mode='sparse',           # match model compile
    shuffle=False                  # Important: keep order for metrics
)

# Predict
preds = model.predict(test_generator)
y_pred = np.argmax(preds, axis=1)
y_true = test_generator.classes

# Metrics
class_labels = list(test_generator.class_indices.keys())
print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_labels, zero_division=0))

print("Confusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))
