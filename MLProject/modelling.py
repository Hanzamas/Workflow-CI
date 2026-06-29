"""
modelling.py — Workflow-CI/MLProject/modelling.py
Dijalankan oleh GitHub Actions via: mlflow run MLProject/
CATATAN: Jangan panggil mlflow.set_experiment() atau mlflow.start_run()
         karena MLflow CLI sudah membuat run aktif secara otomatis.
"""

import os
import argparse
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE    = (64, 64)
BATCH_SIZE  = 32
NUM_CLASSES = 5


def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ], name='rice_cnn_ci')
    return model


def main(data_path: str, epochs: int):
    print(f"[CI] Training rice classification model...")
    print(f"[CI] Data path : {data_path}")
    print(f"[CI] Epochs    : {epochs}")

    # autolog saja — JANGAN set_experiment() atau start_run()
    # mlflow run CLI sudah membuat active run otomatis
    mlflow.tensorflow.autolog(log_models=False)

    train_dir = os.path.join(data_path, "train")
    val_dir   = os.path.join(data_path, "val")

    train_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=True
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=False
    )

    # Training langsung — run sudah aktif dari CLI
    model = build_model((*IMG_SIZE, 3), NUM_CLASSES)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)

    # Log params dan artifacts ke run aktif
    mlflow.log_params({
        "data_path"  : data_path,
        "epochs"     : epochs,
        "img_size"   : str(IMG_SIZE),
        "batch_size" : BATCH_SIZE,
    })

    # Simpan model
    model.save("rice_model_ci.keras")
    mlflow.log_artifact("rice_model_ci.keras")

    val_acc = history.history['val_accuracy'][-1]
    mlflow.log_metric("final_val_accuracy", val_acc)
    print(f"[CI] Done — val_accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data_split")
    parser.add_argument("--epochs",    type=int, default=2)
    args = parser.parse_args()
    main(args.data_path, args.epochs)
