import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load your trained model
model = tf.keras.models.load_model("emotion_cnn_model_v2_resumed.keras")

# Define class labels (make sure these are in the same order as your train directory)
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Path to your test image (replace with your own file)
img_path = "/Users/salonikarekar/Desktop/College work/sem 5/sentiment analysis project files /204.jpg"

# Load and preprocess the image in grayscale
img = image.load_img(img_path, target_size=(48, 48), color_mode='grayscale')
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # normalize same as during training

# Predict
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
predicted_label = class_labels[predicted_class]

# Show result
plt.imshow(img, cmap='gray')
plt.title(f"Predicted emotion: {predicted_label}")
plt.axis("off")
plt.show()
