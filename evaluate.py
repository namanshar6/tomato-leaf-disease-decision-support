"""
evaluate.py
-----------
Implements the "Model Evaluation" module from the architecture diagram:
computes accuracy, precision, recall, and F1-score on the held-out
validation split, and saves a confusion matrix plot.

Usage:
    python evaluate.py
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--model_path", default="model/tomato_disease_model.h5")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    with open("model/class_indices.json") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_gen = val_datagen.flow_from_directory(
        f"{args.data_dir}/val",
        target_size=IMG_SIZE,
        batch_size=args.batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    model = tf.keras.models.load_model(args.model_path)
    preds = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\nAccuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f} (macro)")
    print(f"Recall:    {recall:.4f} (macro)")
    print(f"F1-score:  {f1:.4f} (macro)\n")

    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — Tomato Leaf Disease Classification")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("model/confusion_matrix.png")
    print("Saved confusion matrix to model/confusion_matrix.png")

    # Save a summary metrics file — handy to show in your review
    with open("model/eval_metrics.json", "w") as f:
        json.dump(
            {"accuracy": acc, "precision_macro": precision, "recall_macro": recall, "f1_macro": f1},
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
