import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks, optimizers
import os
import shutil
import numpy as np

#  PATHS 
ck_dir = r"/Users/salonikarekar/Desktop/College work/sem 5/sentiment analysis project files /CK+"
base_dir = r"/Users/salonikarekar/Desktop/College work/sem 5/sentiment analysis project files /CK_split"

train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

# SPLIT 80/20
def split_ck_dataset(src, dst_train, dst_val, split=0.8):

    if os.path.exists(dst_train):
        shutil.rmtree(dst_train)
    if os.path.exists(dst_val):
        shutil.rmtree(dst_val)

    os.makedirs(dst_train)
    os.makedirs(dst_val)

    for cls in os.listdir(src):
        cls_path = os.path.join(src, cls)
        if not os.path.isdir(cls_path):
            continue

        images = os.listdir(cls_path)
        np.random.shuffle(images)

        train_count = int(len(images) * split)

        train_cls_path = os.path.join(dst_train, cls)
        val_cls_path = os.path.join(dst_val, cls)

        os.makedirs(train_cls_path, exist_ok=True)
        os.makedirs(val_cls_path, exist_ok=True)

        # copy files
        for img in images[:train_count]:
            shutil.copy(os.path.join(cls_path, img), train_cls_path)

        for img in images[train_count:]:
            shutil.copy(os.path.join(cls_path, img), val_cls_path)

    print("Dataset created!")
    print("Train Path:", dst_train)
    print("Validation Path:", dst_val)

split_ck_dataset(ck_dir, train_dir, val_dir)

#IMAGE GENERATORS
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

train_gen = ImageDataGenerator(
    rescale=1/255.,
    rotation_range=15,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
).flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse'
)

val_gen = ImageDataGenerator(
    rescale=1/255.
).flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse'
)

#  MobileNetV2 Model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4),
    layers.Dense(7, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# TRAIN (head-only)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5
)

#  FINE-TUNE LAST 30 LAYERS

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_ft = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

model.save("ckplus_mobilenetv2.keras")
print("MODEL SAVED!")
