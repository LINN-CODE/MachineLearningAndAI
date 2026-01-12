import os
import shutil
import splitfolders
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import save_img

# ==========================================
# 1. AUTO-ORGANIZE & CREATE UNKNOWN CLASS
# ==========================================
raw_dir = "raw_data"
processed_dir = "processed_dataset"
# Add "unknown" to your list of targets
target_folders = ["object_1", "object_2", "unknown"] 

if not os.path.exists(raw_dir):
    os.makedirs(raw_dir)
    print(f"Created '{raw_dir}' folder.")

# Move existing object folders if they are in the root
for folder in target_folders:
    if os.path.exists(folder):
        print(f"Moving '{folder}' into '{raw_dir}'...")
        shutil.move(folder, os.path.join(raw_dir, folder))

# --- NEW STEP: Ensure 'unknown' folder exists and has data ---
unknown_path = os.path.join(raw_dir, "unknown")
if not os.path.exists(unknown_path):
    os.makedirs(unknown_path)
    print("Created 'unknown' folder.")

# Generate Dummy Noise Images if 'unknown' is empty
# (Prevents crash. Replace these with REAL background photos later!)
if not os.listdir(unknown_path):
    print("⚠️ 'unknown' folder is empty! Generating 50 dummy noise images...")
    for i in range(50):
        # Create random noise image (224x224)
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype='uint8')
        save_img(os.path.join(unknown_path, f"noise_{i}.jpg"), img_array)
    print("✅ Dummy images created. PLEASE ADD REAL BACKGROUND PHOTOS LATER.")

# ==========================================
# 2. SPLIT DATA (Force Update)
# ==========================================
# If we added a new class, we must delete the old split to force a refresh
if os.path.exists(processed_dir):
    # Check if the existing split has all 3 classes
    existing_classes = os.listdir(os.path.join(processed_dir, "train"))
    if len(existing_classes) < len(target_folders):
        print("New class detected! Deleting old split to refresh...")
        shutil.rmtree(processed_dir)

print("\nSplitting images into Train (80%), Val (10%), Test (10%)...")
splitfolders.ratio(raw_dir, output=processed_dir, 
                   seed=42, ratio=(.8, .1, .1), group_prefix=None)
print("✅ Data split successfully!")

# ==========================================
# 3. GPU CHECK
# ==========================================
print(f"\nTensorFlow Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ SUCCESS! Found GPU: {gpus[0].name}")
else:
    print("⚠️ WARNING: No GPU found. Training might be slow.")

# ==========================================
# 4. LOAD DATASET & CALC WEIGHTS
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
print(f"Classes found: {class_names}") # Should now say ['object_1', 'object_2', 'unknown']

# Calculate Class Weights (Crucial for balancing)
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

class_weights = {}
for i, cls in enumerate(class_names):
    if class_counts[cls] > 0:
        weight = total_samples / (2 * class_counts[cls])
        class_weights[i] = weight
    else:
        class_weights[i] = 1.0

print(f"✅ Weights Applied: {class_weights}")

# Optimize
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# ==========================================
# 5. BUILD MODEL (With Augmentation)
# ==========================================
learning_rate = 1e-5
l2_strength = 1e-4 

# Augmentation helps significantly with small "unknown" datasets
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

print("\nBuilding Model...")
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    data_augmentation,
    layers.Rescaling(1./255),
    
    layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
    layers.Dropout(0.5),
    
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    
    # OUTPUT LAYER: Automatically adjusts to 3 classes
    layers.Dense(len(class_names)) 
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# ==========================================
# 6. TRAIN
# ==========================================
print("\nStarting Training...")
epochs = 50
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    class_weight=class_weights # Apply balancing
)

# ==========================================
# 7. SAVE & PLOT
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
plt.show()