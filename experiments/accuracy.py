import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

model = tf.keras.models.load_model("emotion_cnn_model.h5")

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # works with integer labels
    metrics=['accuracy']
)

class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "/Users/salonikarekar/Desktop/College work/sem 5/sentiment analysis project files /image_sentiment_FER_Dataset/test",   # <-- Replace with your test folder path
    image_size=(48, 48),
    color_mode='rgb',  # matches your trained model input
    batch_size=32,
    shuffle=False
)

loss, accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

#  Evaluate training dataset (quick check for overfitting)
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "/Users/salonikarekar/Desktop/College work/sem 5/sentiment analysis project files /image_sentiment_FER_Dataset/train",
    image_size=(48,48),
    color_mode='rgb',
    batch_size=32,
    shuffle=False
)
train_loss, train_acc = model.evaluate(train_dataset)
print(f"Train Accuracy: {train_acc:.4f}")

y_pred = []# Predict all images in the dataset at once
y_pred = np.argmax(model.predict(test_dataset), axis=1)

# Get true labels
y_true = np.concatenate([y for x, y in test_dataset], axis=0)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))


# ----------------------------
img_path = "/Users/salonikarekar/Desktop/College work/sem 5/sentiment analysis project files /my_img/test_img1.jpg"
img = image.load_img(img_path, target_size=(48, 48))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # normalize

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
predicted_label = class_labels[predicted_class]

plt.imshow(img)
plt.title(f"Predicted emotion: {predicted_label}")
plt.axis("off")
plt.show()
