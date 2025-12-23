# evaluating mobilenetv2_finetuned.keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load your trained model
model = tf.keras.models.load_model("mobilenetv2_finetuned.keras")

# Class labels (must match your dataset folder names)
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Test data generator
test_dir = "/Users/salonikarekar/Desktop/College work/sem 5/sentiment analysis project files /image_sentiment_FER_Dataset/test"
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48,48),
    color_mode='rgb',   # MobileNetV2 expects 3 channels
    batch_size=32,
    class_mode='sparse',
    shuffle=False
)

# Predictions
preds = model.predict(test_generator, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = test_generator.classes

# Metrics
print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_labels))

print("Confusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))
