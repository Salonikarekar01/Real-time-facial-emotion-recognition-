import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.makedirs("confusion_matrices", exist_ok=True)
os.makedirs("comparison_charts", exist_ok=True) 
# ---------- CONFIG ----------
TEST_DIR = "/Users/salonikarekar/Desktop/College work/sem 5/sentiment analysis project files /archive (3)/DATASET/test"
CLASS_NAMES = ["1","2","3","4","5","6","7"]

MODEL_PATHS = {
    "CKPLUS_MobileNetV2": "ckplus_mobilenetv2.keras",
    "RAFDB_MobileNetV2": "mobilenetv2_rafdb_finetuned.keras",
    "Emotion_CNN_Model_V2": "emotion_cnn_model_v2_resumed.keras",
    # add more models here if needed
}

os.makedirs("confusion_matrices", exist_ok=True)
os.makedirs("comparison_charts", exist_ok=True)

# ---------- HELPER FUNCTIONS ----------

def get_model_input_details(model):
    """Return input shape of a model as (height, width, channels)."""
    try:
        shape = model.input.shape
    except:
        shape = model.inputs[0].shape
    return int(shape[1]), int(shape[2]), int(shape[3])

def get_inference_time(model, img_shape):
    """Return time in ms for a single dummy prediction."""
    dummy = np.random.random(img_shape).astype("float32")
    t0 = time.time()
    model.predict(dummy)
    t1 = time.time()
    return (t1 - t0) * 1000

def get_model_size(path):
    """Return model size in MB."""
    size = os.path.getsize(path) / (1024 * 1024)
    return round(size, 2)

def plot_bar_chart(values_dict, metric, ylabel, save_name):
    models = list(values_dict.keys())
    values = [values_dict[m][metric] * 100 if metric in ["accuracy", "f1", "precision", "recall"] else values_dict[m][metric] for m in models]
    
    plt.figure(figsize=(8,5))
    plt.bar(models, values)
    plt.title(f"{metric.capitalize()} Comparison")
    plt.ylabel(ylabel)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"comparison_charts/{save_name}.png")
    plt.close()

# ---------- MAIN LOOP ----------
results = {}

for name, path in MODEL_PATHS.items():
    print(f"\nEvaluating: {name}\n")
    
    # Load model
    try:
        model = load_model(path)
    except Exception as e:
        print(f"Cannot load model {path}: {e}")
        continue

    img_h, img_w, channels = get_model_input_details(model)
    target_size = (img_h, img_w)
    color_mode = "rgb" if channels == 3 else "grayscale"

    print(f"Detected input: size={target_size}, channels={channels}, mode={color_mode}")

    # Test data generator
    datagen = ImageDataGenerator(rescale=1/255.)
    test_gen = datagen.flow_from_directory(
        TEST_DIR,
        target_size=target_size,
        color_mode=color_mode,
        batch_size=1,
        shuffle=False,
        classes=CLASS_NAMES
    )

    y_true = test_gen.classes

    # Prediction
    y_pred_prob = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    inference_ms = get_inference_time(model, (1, img_h, img_w, channels))
    size = get_model_size(path)

    # Store in results
    results[name] = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "inference_ms": inference_ms,
        "size": size
    }

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"confusion_matrices/{name}_cm.png")
    plt.close()
    print(f"Saved confusion matrix: confusion_matrices/{name}_cm.png")

print("\nAll models evaluated successfully!")

# ---------- PLOTTING COMPARISONS ----------
plot_bar_chart(results, "accuracy", "Accuracy (%)", "accuracy_comparison")
plot_bar_chart(results, "f1", "F1 Score (%)", "f1_comparison")
plot_bar_chart(results, "precision", "Precision (%)", "precision_comparison")
plot_bar_chart(results, "recall", "Recall (%)", "recall_comparison")
plot_bar_chart(results, "inference_ms", "Time (ms)", "inference_time")
plot_bar_chart(results, "size", "Size (MB)", "model_sizes")
