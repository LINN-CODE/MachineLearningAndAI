import os
import shutil
import splitfolders
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. AUTO-ORGANIZE FOLDERS
# ==========================================
raw_dir = "raw_data"
processed_dir = "processed_dataset"
target_folders = ["object_1", "object_2"] # Make sure these match your actual folder names!

if not os.path.exists(raw_dir):
    os.makedirs(raw_dir)

for folder in target_folders:
    if os.path.exists(folder):
        print(f"Moving '{folder}' into '{raw_dir}'...")
        shutil.move(folder, os.path.join(raw_dir, folder))

# Only split if we haven't done it yet (saves time on re-runs)
if not os.path.exists(processed_dir):
    print("\nSplitting images into Train (80%), Val (10%), Test (10%)...")
    splitfolders.ratio(raw_dir, output=processed_dir, 
                       seed=42, ratio=(.8, .1, .1), group_prefix=None)
    print("✅ Data split successfully!")
else:
    print("✅ Data already split. Skipping split step.")

# ==========================================
# 2. GPU CHECK
# ==========================================
print(f"\nTensorFlow Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ SUCCESS! Found GPU: {gpus[0].name}")
else:
    print("⚠️ WARNING: No GPU found. Training might be slow.")

# ==========================================
# 3. LOAD DATASET
# ==========================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32 

print("\nLoading datasets...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{processed_dir}/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    f"{processed_dir}/val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print(f"Classes found: {class_names}")

# ==========================================
# 4. HANDLE IMBALANCE (Auto-Calculate Weights)
# ==========================================
# This section counts your files and calculates weights.
# If Dragonfruit is rare, it gets a HUGE weight.
print("\nCalculating Class Weights...")
train_dir = os.path.join(processed_dir, "train")
class_counts = {}
total_samples = 0

for cls in class_names:
    cls_path = os.path.join(train_dir, cls)
    count = len(os.listdir(cls_path))
    class_counts[cls] = count
    total_samples += count
    print(f"   -> {cls}: {count} images")

# Formula: Weight = Total / (2 * Count)
# Rare classes get higher numbers. Common classes get lower numbers.
class_weights = {}
for i, cls in enumerate(class_names):
    weight = total_samples / (2 * class_counts[cls])
    class_weights[i] = weight

print(f"✅ Weights Applied: {class_weights}")
print("(Rare classes will have higher weights to force learning)")

# Optimization
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# ==========================================
# 5. AUGMENTATION & MODEL
# ==========================================
learning_rate = 1e-5
l2_strength = 1e-4

# Step A: Define Augmentation Layer
# This randomly flips/rotates images to "create" more data for the rare class
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2), # Rotate up to 20%
    layers.RandomZoom(0.2),     # Zoom up to 20%
    layers.RandomContrast(0.2), # Adjust lighting
])

print("\nBuilding Model...")
model = models.Sequential([
    # Input
    layers.Input(shape=(224, 224, 3)),
    
    # 1. ADD AUGMENTATION HERE
    data_augmentation,
    
    # Rescaling
    layers.Rescaling(1./255),
    
    # CNN Blocks
    layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    
    # Dense Layers
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
    layers.Dropout(0.5),
    
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    
    layers.Dense(len(class_names))
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# ==========================================
# 6. TRAIN (With Class Weights)
# ==========================================
print("\nStarting Training...")

epochs = 50
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    class_weight=class_weights # <--- THIS IS THE KEY FIX FOR IMBALANCE
)

# ==========================================
# 7. VISUALIZATION
# ==========================================
model.save('my_model.h5')
print("\n✅ Model saved as 'my_model.h5'")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, 'bo', label='Training acc')
plt.plot(epochs_range, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, 'bo', label='Training loss')
plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc='upper right')

plt.savefig('advanced_analysis.png')
print("✅ Graphs saved as 'advanced_analysis.png'")
plt.show()