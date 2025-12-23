import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

TEST_DIR = "/Users/salonikarekar/Desktop/College work/sem 5/sentiment analysis project files /archive (3)/DATASET/test"
CLASS_NAMES = ["1","2","3","4","5","6","7"]

MODEL_PATHS = {
    "CKPLUS_MobileNetV2"        : "ckplus_mobilenetv2.keras",
    "RAFDB_MobileNetV2"         : "mobilenetv2_rafdb_finetuned.keras",
    "Emotion_CNN_Model_V2"      : "emotion_cnn_model_v2_resumed.keras"
}

os.makedirs("confusion_matrices", exist_ok=True)

# Detect input size & channels
def get_model_input_details(model):
    try:
        shape = model.input.shape        # works for most models
    except:
        shape = model.inputs[0].shape    # fallback for Sequential without input called

    height = int(shape[1])
    width = int(shape[2])
    channels = int(shape[3])
    return (height, width, channels)

# MAIN LOOP
for name, path in MODEL_PATHS.items():
    print("\n")
    print(f" Evaluating: {name}")
    print("\n")

    try:
        model = load_model(path)
    except:
        print(f" Cannot load model: {path}")
        continue

    # detect input shape
    img_h, img_w, channels = get_model_input_details(model)
    target_size = (img_h, img_w)
    color_mode = "rgb" if channels == 3 else "grayscale"

    print(f" Detected input: size={target_size}, channels={channels}, mode={color_mode}")

    # test generator
    datagen = ImageDataGenerator(rescale=1/255.)
    test_gen = datagen.flow_from_directory(
        TEST_DIR,
        target_size=target_size,
        color_mode=color_mode,
        batch_size=1,
        shuffle=False,
        classes=CLASS_NAMES
    )

    # ground truth labels
    y_true = test_gen.classes

    # Predict
    try:
        y_pred_prob = model.predict(test_gen, verbose=1)
    except Exception as e:
        print(f" Prediction failed for {name} → {e}")
        continue

    y_pred = np.argmax(y_pred_prob, axis=1)

    # accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f" Accuracy for {name}: {acc*100:.2f}%")

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    save_path = f"confusion_matrices/{name}_cm.png"
    plt.savefig(save_path)
    plt.close()

    print(f" Saved confusion matrix: {save_path}")

print("\n All models evaluated successfully!")

results = {
    "CKPLUS_MobileNetV2": {"accuracy": 0.1033},
    "RAFDB_MobileNetV2": {"accuracy": 0.7405},
    "Emotion_CNN_Model_V2": {"accuracy": 0.3680}
}

# Create folder if it doesn't exist
os.makedirs("comparison_charts", exist_ok=True)

 #accuracy comparison 

def plot_accuracy_bar(results):
    models = list(results.keys())
    accuracies = [results[m]["accuracy"] * 100 for m in models]

    plt.figure(figsize=(8,5))
    plt.bar(models, accuracies)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("comparison_charts/accuracy_comparison.png")
    plt.close()

plot_accuracy_bar(results)

#Inference Time (ms/frame) Graph
def get_inference_time(model, img_shape):
    dummy = np.random.random(img_shape).astype("float32")
    t0 = time.time()
    model.predict(dummy)
    t1 = time.time()
    return (t1 - t0) * 1000   # in ms

results[name]["inference_ms"] = get_inference_time(model, (1, img_h, img_w, channels))

#plotting
def plot_inference_time(results):
    models = list(results.keys())
    times = [results[m]["inference_ms"] for m in models]

    plt.figure(figsize=(8,5))
    plt.bar(models, times)
    plt.title("Inference Time per Model")
    plt.ylabel("Time (ms)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("comparison_charts/inference_time.png")
    plt.close()

plot_inference_time(results)

 #model size comparison

def get_model_size(path):
    size = os.path.getsize(path) / (1024 * 1024)
    return round(size, 2)

results["CKPLUS_MobileNetV2"]["size"] = get_model_size("ckplus_mobilenetv2.keras")
results["RAFDB_MobileNetV2"]["size"] = get_model_size("mobilenetv2_rafdb_finetuned.keras")
results["Emotion_CNN_Model_V2"]["size"] = get_model_size("emotion_cnn_model_v2_resumed.keras")

def plot_model_size(results):
    models = list(results.keys())
    sizes = [results[m]["size"] for m in models]

    plt.figure(figsize=(8,5))
    plt.bar(models, sizes)
    plt.title("Model Size Comparison")
    plt.ylabel("Size (MB)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("comparison_charts/model_sizes.png")
    plt.close()

plot_model_size(results)

#precision recall f1 score comparison 

results[name]["precision"] = precision_score(y_true, y_pred, average='macro')
results[name]["recall"] = recall_score(y_true, y_pred, average='macro')
results[name]["f1"] = f1_score(y_true, y_pred, average='macro')

def plot_f1(results):
    models = list(results.keys())
    f1s = [results[m]["f1"] * 100 for m in models]

    plt.figure(figsize=(8,5))
    plt.bar(models, f1s)
    plt.title("F1-Score Comparison")
    plt.ylabel("F1 Score (%)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("comparison_charts/f1_comparison.png")
    plt.close()

plot_f1(results)

#folder creation 
os.makedirs("comparison_charts", exist_ok=True)
