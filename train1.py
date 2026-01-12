import os
import shutil
import splitfolders
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt

# ==========================================
# 1. AUTO-ORGANIZE FOLDERS
# ==========================================
# This section fixes the folder structure automatically based on your screenshot.
# It moves 'object_1' and 'object_2' into a 'raw_data' folder.

raw_dir = "raw_data"
processed_dir = "processed_dataset"
target_folders = ["object_1", "object_2"]

if not os.path.exists(raw_dir):
    os.makedirs(raw_dir)
    print(f"Created '{raw_dir}' folder.")

# Check if folders exist in the root and move them
for folder in target_folders:
    if os.path.exists(folder):
        print(f"Moving '{folder}' into '{raw_dir}'...")
        shutil.move(folder, os.path.join(raw_dir, folder))

# Verify if data is ready
if not os.listdir(raw_dir):
    print(f"❌ ERROR: Your '{raw_dir}' folder is empty!")
    print("Please make sure 'object_1' and 'object_2' are inside 'raw_data'.")
    exit()

# ==========================================
# 2. SPLIT DATA (Train/Val/Test)
# ==========================================
print("\nSplitting images into Train (80%), Val (10%), Test (10%)...")

# This creates the 'processed_dataset' folder with train/val/test inside
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
# 4. LOAD DATASET
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

# Optimization: Cache data in memory for faster training
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


# ==========================================
# 5. BUILD CNN MODEL
# ==========================================

learning_rate = 1e-5  # Very small steps for better precision
l2_strength = 1e-4  

print("\nBuilding Model...")

model = models.Sequential([
    # Input & Rescaling
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    
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
    
    # Dense Layers with DROPOUT (The key to the "Best Model")
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
    layers.Dropout(0.5), # Discards 50% of neurons randomly to force learning
    
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3), # Discards 30% of neurons
    
    layers.Dense(len(class_names)) # Output Layer
])

# Use the specific Adam optimizer with the custom learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

# ==========================================
# 6. TRAIN MODEL
# ==========================================
print("\nStarting Training...")

epochs = 50
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# ==========================================
# 7. SAVE RESULTS
# ==========================================
# Save the trained model file
model.save('my_model.h5')
print("\n✅ Model saved as 'my_model.h5'")

# Plot and save the accuracy graph
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(14, 5)) # Wide figure for two graphs

# Graph 1: Accuracy (Left Side)
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, 'bo', label='Training acc') # 'bo' = Blue Dot
plt.plot(epochs_range, val_acc, 'b', label='Validation acc') # 'b' = Blue Line
plt.title('Training and validation accuracy')
plt.legend(loc='lower right')

# Graph 2: Loss (Right Side)
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, 'bo', label='Training loss')
plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc='upper right')

# Save and Show
plt.savefig('advanced_analysis.png')
print("✅ Graphs saved as 'advanced_analysis.png'")
plt.show()