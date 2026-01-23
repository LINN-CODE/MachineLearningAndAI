import os
import shutil
import splitfolders
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as plt

# ==========================================
# ==========================================
# This speeds up training on RTX cards by using 16-bit math where possible
tf.keras.mixed_precision.set_global_policy('mixed_float16')

print(f"TensorFlow Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ SUCCESS! Found GPU: {gpus[0].name}")
else:
    print("⚠️ WARNING: No GPU found. Training might be slow.")

# ==========================================
# 1. AUTO-ORGANIZE & CLEANUP
# ==========================================
raw_dir = "raw_data"
processed_dir = "processed_dataset"
# IMPORTANT: Ensure your folders in 'raw_data' match these names exactly
target_classes = ["banana", "dragonfruit", "unknown"] 

# Safety check: Remove old processed data to ensure a fresh split
if os.path.exists(processed_dir):
    print(f"Removing old '{processed_dir}' to ensure fresh split...")
    shutil.rmtree(processed_dir)

# Check if raw data exists
if not os.path.exists(raw_dir) or not os.listdir(raw_dir):
    print(f"❌ ERROR: '{raw_dir}' is missing or empty!")
    print(f"Please create '{raw_dir}' and put folders {target_classes} inside.")
    exit()

# ==========================================
# 2. SPLIT DATA (Train/Val/Test)
# ==========================================
print("\nSplitting images into Train (80%), Val (10%), Test (10%)...")

# Ratio: 80% Train, 10% Validation, 10% Test
splitfolders.ratio(raw_dir, output=processed_dir, 
                   seed=1337, ratio=(.8, .1, .1), group_prefix=None)

print("✅ Data split successfully!")

# ==========================================
# 3. LOAD DATASET
# ==========================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 16 

print("\nLoading datasets...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{processed_dir}/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    f"{processed_dir}/val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print(f"Classes found: {class_names}")

# Optimization: Cache data in memory for speed
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# ==========================================
# 4. AGGRESSIVE AUGMENTATION (The Anti-Color Bias Fix)
# ==========================================
data_augmentation = tf.keras.Sequential([
    # --- Geometric (Shape) ---
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.1, 0.1),

        
    # Standard lighting changes
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
])

# ==========================================
# 5. BUILD CNN MODEL
# ==========================================
learning_rate = 1e-4  # Keep low for transfer learning
l2_strength = 1e-4    

print("\nDownloading and Building EfficientNetB0...")

# 1. Load the base model (Pre-trained on ImageNet)
# include_top=False removes the 1000-class classification layer
base_model = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)

# 2. Freeze the base model
# This ensures we only train your new top layers first
base_model.trainable = False

# 3. Create the final model
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    
    # Augmentation (Keep your existing strong augmentation)
    data_augmentation,
    
    # NOTE: We REMOVED layers.Rescaling(1./255) 
    # EfficientNet expects 0-255 range inputs!
    
    # The Pre-trained Base
    base_model,
    
    # Convert features to a single vector per image
    layers.GlobalAveragePooling2D(),
    
    # Dropout to prevent overfitting
    layers.Dropout(0.3),
    
    # Output Layer
    layers.Dense(len(class_names)) # No softmax (logits=True in loss)
])

# Use separate definition for loss/optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=['accuracy'])

model.summary()

# ==========================================
# 6. TRAIN
# ==========================================
print("\nStarting Training...")

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=8,          # Stop if no improvement for 8 epochs
    restore_best_weights=True,
    verbose=1
)

# Reduce Learning Rate if we get stuck (helps fine-tune accuracy)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=4, 
    min_lr=1e-6,
    verbose=1
)

epochs = 50 

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[early_stopping, reduce_lr]
)

# ==========================================
# 7. SAVE RESULTS
# ==========================================
model.save('Jan_23_EfficientNet_model.h5') 
print("\n✅ Model saved as 'Jan21_balance_model.h5'")

# Plotting
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, 'bo-', label='Training acc')
plt.plot(epochs_range, val_acc, 'r-', label='Validation acc')
plt.title('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, 'bo-', label='Training loss')
plt.plot(epochs_range, val_loss, 'r-', label='Validation loss')
plt.title('Loss')
plt.legend(loc='upper right')
plt.grid(True)

plt.savefig('training_graph2.png')
print("✅ Graphs saved as 'training_graph.png'")
plt.show()