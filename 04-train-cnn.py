import os
import shutil
import pandas as pd
import numpy as np

# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

print('TensorFlow version:', tf.__version__)

# --- Dataset Paths ---
dataset_path = './split_dataset/'
tmp_debug_path = './tmp_debug'
checkpoint_path = './tmp_checkpoint'

for path in [tmp_debug_path, checkpoint_path]:
    print(f'Creating Directory: {path}')
    os.makedirs(path, exist_ok=True)

def get_filename_only(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

# --- Image Data Generators ---
input_size = 128
batch_size_num = 32

train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')
test_path = os.path.join(dataset_path, 'test')

train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    directory=train_path,
    target_size=(input_size, input_size),
    color_mode="rgb",
    class_mode="binary",
    batch_size=batch_size_num,
    shuffle=True
)

val_datagen = ImageDataGenerator(rescale=1/255)

val_generator = val_datagen.flow_from_directory(
    directory=val_path,
    target_size=(input_size, input_size),
    color_mode="rgb",
    class_mode="binary",
    batch_size=batch_size_num,
    shuffle=True
)

test_datagen = ImageDataGenerator(rescale=1/255)

test_generator = test_datagen.flow_from_directory(
    directory=test_path,
    classes=['real', 'fake'],
    target_size=(input_size, input_size),
    color_mode="rgb",
    class_mode=None,
    batch_size=1,
    shuffle=False
)

# --- Build CNN Model with TensorFlow EfficientNetB0 ---
efficient_net = EfficientNetB0(
    include_top=False,
    weights='imagenet',  # Uses TF maintained pretrained weights
    input_shape=(input_size, input_size, 3),
    pooling='max'
)

model = Sequential([
    efficient_net,
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.summary()

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# --- Callbacks ---
custom_callbacks = [
    EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=5,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(checkpoint_path, 'best_model.h5'),
        monitor='val_loss',
        mode='min',
        verbose=1,
        save_best_only=True
    )
]

# --- Train Model ---
num_epochs = 20
history = model.fit(
    train_generator,
    epochs=num_epochs,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=custom_callbacks
)

print("✅ Training Completed")
print(history.history)

# --- Load Best Model ---
best_model = load_model(os.path.join(checkpoint_path, 'best_model.h5'))

# --- Generate Predictions on Test Set ---
test_generator.reset()
preds = best_model.predict(test_generator, verbose=1)

test_results = pd.DataFrame({
    "Filename": test_generator.filenames,
    "Prediction": preds.flatten()
})

print(test_results)

# Optional: Save predictions
test_results.to_csv('test_predictions.csv', index=False)
print("✅ Predictions saved to test_predictions.csv")
