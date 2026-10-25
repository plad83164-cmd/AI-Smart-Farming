import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# -----------------------------
# SETTINGS
# -----------------------------

DATASET_PATH = "dataset/Tomato"

IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 25

# -----------------------------
# DATA AUGMENTATION
# -----------------------------

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

# -----------------------------
# LOAD TRAINING DATA
# -----------------------------

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42
)

# -----------------------------
# LOAD VALIDATION DATA
# -----------------------------

validation_data = validation_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# -----------------------------
# CNN MODEL
# -----------------------------

model = models.Sequential([

    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(256, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(
        train_data.num_classes,
        activation="softmax"
    )
])

# -----------------------------
# COMPILE
# -----------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# CALLBACKS
# -----------------------------

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.00001
)

# -----------------------------
# SHOW MODEL
# -----------------------------

model.summary()

# -----------------------------
# TRAIN
# -----------------------------

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        reduce_lr
    ]
)

# -----------------------------
# SAVE CLASS NAMES
# -----------------------------

class_names = list(train_data.class_indices.keys())

with open("class_names.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

# -----------------------------
# SAVE MODEL
# -----------------------------

model.save("tomato_disease_model.keras")

print("\nTraining completed successfully!")
print("Model saved as: tomato_disease_model.keras")

print("\nClass names:")
print(class_names)