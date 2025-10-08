import tensorflow as tf
import os

# === Path ng dataset folder ===
DATASET_DIR = "dataset"

# === Create datasets ===
def make_datasets(img_size=(128, 128), batch_size=16):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(DATASET_DIR, "train"),
        image_size=img_size,
        batch_size=batch_size
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(DATASET_DIR, "val"),
        image_size=img_size,
        batch_size=batch_size
    )

    # ✅ Kunin muna class names bago i-map
    class_names = train_ds.class_names
    print("Classes:", class_names)

    # Normalize (0-1)
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    return train_ds, val_ds, class_names


# === Build model ===
def make_model(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# === Main training ===
def main():
    train_ds, val_ds, class_names = make_datasets()

    model = make_model(num_classes=len(class_names))

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10
    )

    # === Save model + class names ===
    model.save("egg_model.keras")
    with open("class_names.txt", "w") as f:
        for name in class_names:
            f.write(name + "\n")

    print("✅ Training complete! Model saved as egg_model.keras")


if __name__ == "__main__":
    main()
