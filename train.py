"""
train.py
--------
Implements the "CNN / Transfer Learning Model" training path from the
architecture diagram. Trains a MobileNetV2-based transfer-learning model
on the tomato leaf dataset (data/train, data/val produced by data_prep.py).

Usage:
    python train.py --epochs 15 --batch_size 32
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)


def build_model(num_classes: int) -> tf.keras.Model:
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Freeze pretrained backbone initially

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fine_tune_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_out", default="model/tomato_disease_model.h5")
    args = parser.parse_args()

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=args.batch_size, class_mode="categorical"
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir, target_size=IMG_SIZE, batch_size=args.batch_size, class_mode="categorical", shuffle=False
    )

    num_classes = len(train_gen.class_indices)
    print(f"Detected {num_classes} classes: {train_gen.class_indices}")

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    with open("model/class_indices.json", "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)

    model, base_model = build_model(num_classes)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.model_out, save_best_only=True, monitor="val_accuracy"),
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

    print("\n=== Phase 1: training classification head (backbone frozen) ===")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    print("\n=== Phase 2: fine-tuning top layers of MobileNetV2 backbone ===")
    base_model.trainable = True
    # Freeze all but the last ~30 layers so fine-tuning is fast and stable
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.fine_tune_epochs,
        callbacks=callbacks,
    )

    model.save(args.model_out)
    print(f"\nSaved final model to {args.model_out}")

    # Plot combined training curves
    acc = history1.history["accuracy"] + history2.history["accuracy"]
    val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
    loss = history1.history["loss"] + history2.history["loss"]
    val_loss = history1.history["val_loss"] + history2.history["val_loss"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(acc, label="train acc")
    axes[0].plot(val_acc, label="val acc")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(loss, label="train loss")
    axes[1].plot(val_loss, label="val loss")
    axes[1].set_title("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("model/training_curves.png")
    print("Saved training curves to model/training_curves.png")


if __name__ == "__main__":
    main()
