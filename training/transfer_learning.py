import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0

# Paths
train_dir = "/Users/salonikarekar/Desktop/College work/sem 5/sentiment analysis project files /image_sentiment_FER_Dataset/train"
test_dir = "/Users/salonikarekar/Desktop/College work/sem 5/sentiment analysis project files /image_sentiment_FER_Dataset/test"

# Data generators with augmentation
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
    target_size=(48,48),
    color_mode='rgb',  # EfficientNetB0 expects 3 channels
    batch_size=32,
    class_mode='sparse'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48,48),
    color_mode='rgb',
    batch_size=32,
    class_mode='sparse'
)
# Load EfficientNetB0 without top layers
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(48,48,3))
base_model.trainable = False  # Freeze the base model initially

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')  # 7 emotion classes
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#training classification head first 
history_head = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=10,  # Start with 10 epochs
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
)


#fine tuning base model layers 
# Unfreeze last few layers of EfficientNetB0
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False  # freeze all except last 20 layers

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),  # lower LR for fine-tuning
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_finetune = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=10,  # you can adjust
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
)
