import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2D, MaxPooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- Configuration ---
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
FINE_TUNE_EPOCHS = 5
DATA_DIR = 'Logos'

# --- Data Generator ---
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Get number of brand classes
num_classes = len(train_generator.class_indices)
print(f"Number of brand classes: {num_classes}")
print(f"Class indices: {train_generator.class_indices}")
with open("class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f, indent=4)
print("Class indices saved!")

# --- Load Pretrained MobileNetV3 ---
base_model = MobileNetV3Small(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                              include_top=False,
                              weights='imagenet')

base_model.trainable = False  # Freeze initially

# --- Custom Top Layers with Conv + MaxPooling + Dropout ---
x = base_model.output
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.3)(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.3)(x)

x = GlobalAveragePooling2D()(x)                  
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)

brand_predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=brand_predictions)

print(f"\nModel Summary:")
print(f"Total parameters: {model.count_params():,}")

# --- Callbacks ---
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint('brand_classifier_mobilenet.keras', monitor='val_accuracy', save_best_only=True, verbose=1)

# --- Compile and Train (Frozen Base) ---
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("\n=== Training Phase 1: Frozen Base Model ===")
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# --- Unfreeze last few layers for Fine-Tuning ---
base_model.trainable = True

# Freeze first layers, unfreeze last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Count trainable parameters
trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
print(f"Trainable parameters for fine-tuning: {trainable_params:,}")

# Compile again with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # Lower LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tune
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# Save final model
model.save('final_brand_classifier_mobilenetv3.keras')
print("\n✅ Brand Classification Model Saved!")