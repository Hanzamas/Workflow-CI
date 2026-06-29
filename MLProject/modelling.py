"""
modelling.py — Workflow-CI/MLProject/modelling.py
Versi CI dari training script — dijalankan oleh GitHub Actions via MLflow Project
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

    mlflow.set_experiment("rice-classification-ci")
    mlflow.tensorflow.autolog(log_models=True)

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

    with mlflow.start_run(run_name="ci_run"):
        model = build_model((*IMG_SIZE, 3), NUM_CLASSES)
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)

        mlflow.log_params({
            "data_path"  : data_path,
            "epochs"     : epochs,
            "img_size"   : str(IMG_SIZE),
            "batch_size" : BATCH_SIZE,
        })

        # Simpan model sebagai artifact
        model.save("rice_model_ci")
        mlflow.log_artifact("rice_model_ci")

        print(f"[CI] Done — val_accuracy: {history.history['val_accuracy'][-1]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="MLProject/data_split")
    parser.add_argument("--epochs",    type=int, default=5)
    args = parser.parse_args()
    main(args.data_path, args.epochs)
