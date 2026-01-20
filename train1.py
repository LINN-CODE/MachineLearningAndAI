import os
import shutil
import splitfolders
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt

# ==========================================
# 1. AUTO-ORGANIZE FOLDERS
# ==========================================
raw_dir = "raw_data"
processed_dir = "processed_dataset"
target_folders = ["object_1", "object_2", "unknown"]

if not os.path.exists(raw_dir):
    os.makedirs(raw_dir)
    print(f"Created '{raw_dir}' folder.")

# Move folders if they are in the root
for folder in target_folders:
    if os.path.exists(folder):
        print(f"Moving '{folder}' into '{raw_dir}'...")
        shutil.move(folder, os.path.join(raw_dir, folder))

# Check if data exists
if not os.listdir(raw_dir):
    print(f"❌ ERROR: Your '{raw_dir}' folder is empty!")
    print("Please make sure 'object_1' and 'object_2' are inside 'raw_data'.")
    exit()

# ==========================================
# 2. SPLIT DATA (Train/Val/Test)
# ==========================================
print("\nSplitting images into Train (80%), Val (10%), Test (10%)...")

# Ratio: 80% Train, 10% Validation, 10% Test
splitfolders.ratio(raw_dir, output=processed_dir, 
                   seed=42, ratio=(.8, .1, .1), group_prefix=None)

print("✅ Data split successfully!")

# ==========================================
# 3. CONFIGURATION & GPU CHECK
# ==========================================
IMG_SIZE = (224, 224)

BATCH_SIZE = 32 

print(f"\nTensorFlow Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ SUCCESS! Found GPU: {gpus[0].name}")
else:
    print("⚠️ WARNING: No GPU found. Training might be slow.")

# ==========================================
# 4. LOAD DATASET
# ==========================================
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

# Optimization: Cache data in memory for speed
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# ==========================================
# 5. DEFINE AUGMENTATION LAYERS
# ==========================================
# These layers only run during training. They are skipped during validation/prediction.
data_augmentation = tf.keras.Sequential([
    # Geometric (Shape/Position)
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),             # Rotate +/- 20%
    layers.RandomZoom(0.2),                 # Zoom +/- 20%
    layers.RandomTranslation(0.1, 0.1),     # Shift position
    
    # Photometric (Lighting/Color)
    layers.RandomBrightness(0.2),           # Darker/Brighter
    layers.RandomContrast(0.2),             # Higher/Lower contrast
])

# ==========================================
# 6. BUILD CNN MODEL
# ==========================================
learning_rate = 1e-5  # Low learning rate for stability
l2_strength = 1e-4    # Regularization to prevent overfitting

print("\nBuilding Model...")

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    
    # Apply Augmentation
    data_augmentation,
    
    # Standardize pixels (0-255 -> 0-1)
    layers.Rescaling(1./255),
    
    # --- Convolutional Blocks ---
    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
    layers.MaxPooling2D((2, 2)),
    
    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
    layers.MaxPooling2D((2, 2)),
    
    # Block 3
    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    
    # --- Dense Layers (Classifier) ---
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
    layers.Dropout(0.5), # 50% Dropout
    
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3), # 30% Dropout
    
    layers.Dense(len(class_names)) # Output
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# ==========================================
# 7. TRAIN WITH EARLY STOPPING
# ==========================================
print("\nStarting Training...")

# Stop if validation loss doesn't improve for 10 epochs
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

epochs = 40 # Maximum epochs (EarlyStopping will likely cut this short)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[early_stopping]
)

# ==========================================
# 8. SAVE RESULTS
# ==========================================
model.save('Jan_20_model.h5')
print("\n✅ Model saved as 'final_augmented_model.h5'")

# Plotting
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Adjust range for actual number of epochs run (in case early stopping hit)
epochs_range = range(len(acc))

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, 'bo', label='Training acc')
plt.plot(epochs_range, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, 'bo', label='Training loss')
plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend(loc='upper right')

plt.savefig('training_graph.png')
print("✅ Graphs saved as 'training_graph.png'")
plt.show()