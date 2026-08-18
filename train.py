"""
Training entry point.

This script fine-tunes a CNN/ResNet-style image classifier on PlantVillage.
For the fastest route, use the public PlantVillage dataset and a GPU environment.

The deployed demo uses the Apache-2.0 ResNet50 checkpoint documented in README.md.
Run this script if you want to produce your own independently trained checkpoint.
"""

from pathlib import Path
import json

import tensorflow as tf
from tensorflow import keras


IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5


def train(data_dir: str = "data/raw"):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"{data_dir} not found. Download PlantVillage first; see data/README.md."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    class_names = train_ds.class_names

    base = keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.25)(x)
    outputs = keras.layers.Dense(len(class_names), activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    Path("model").mkdir(exist_ok=True)
    model.save("model/plant_disease_model.keras")
    Path("model/class_labels.json").write_text(json.dumps(class_names, indent=2))

    print("Saved model/plant_disease_model.keras")
    print("Validation accuracy:", history.history["val_accuracy"][-1])


if __name__ == "__main__":
    train()
