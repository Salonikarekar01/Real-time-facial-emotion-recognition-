import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers, callbacks
import os

# Paths to your dataset
train_dir = "/Users/salonikarekar/Desktop/College work/sem 5/sentiment analysis project files /image_sentiment_FER_Dataset/train"
test_dir  = "/Users/salonikarekar/Desktop/College work/sem 5/sentiment analysis project files /image_sentiment_FER_Dataset/test"

# Image preprocessing
IMG_SIZE = (224, 224)  # MobileNetV2 default input size

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='sparse',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='sparse',
    shuffle=False
)

# Load pre-trained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # freeze base first

# Adding classification head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')  # 7 emotion classes
])

# Compile model (head only)
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
checkpoint_cb = callbacks.ModelCheckpoint(
    "mobilenetv2_head_only.keras",
    save_best_only=True,
    monitor='val_accuracy'
)
earlystop_cb = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True
)

# Train only the head
history_head = model.fit(
    train_generator,
    epochs=5,
    validation_data=test_generator,
    callbacks=[checkpoint_cb, earlystop_cb]
)

# Fine-tune last 30 layers of base
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tune model
history_finetune = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator,
    callbacks=[checkpoint_cb, earlystop_cb]
)

#
model.save("mobilenetv2_finetuned.keras")
print("Fine-tuned MobileNetV2 model saved successfully!")
