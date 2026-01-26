import os
import shutil
import splitfolders
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt

# ==========================================
# 1. SETUP & GPU CHECK
# ==========================================
# We keep Mixed Precision because it makes training fast
tf.keras.mixed_precision.set_global_policy('mixed_float16')

print(f"TensorFlow Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ SUCCESS! Found GPU: {gpus[0].name}")

# ==========================================
# 2. AUTO-ORGANIZE
# ==========================================
raw_dir = "raw_data"
processed_dir = "processed_dataset"

if os.path.exists(processed_dir):
    shutil.rmtree(processed_dir)

# Ensure only valid classes exist
if os.path.exists(os.path.join(raw_dir, "unknown")):
    print("❌ ERROR: Please delete the 'unknown' folder from raw_data first.")
    exit()

print("Splitting data...")
splitfolders.ratio(raw_dir, output=processed_dir, seed=1337, ratio=(.8, .1, .1))

# ==========================================
# 3. LOAD DATASET
# ==========================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 64

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
print(f"🎯 Classes: {class_names}")

# ==========================================
# 4. EXTERNAL AUGMENTATION (The Fix)
# ==========================================
# We define augmentation HERE, but we do NOT put it in the model.
def augment_data(x, y):
    # 1. Geometric Changes
    x = tf.image.random_flip_left_right(x)
    # Rotations are tricky in raw TF, usually handled by layers, 
    # but for safety we will stick to Color/Lighting which caused the crash.
    
    # 2. Color Changes (The crash fix)
    x = tf.image.random_hue(x, 0.1)
    x = tf.image.random_saturation(x, 0.7, 1.3)
    x = tf.image.random_brightness(x, 0.2)
    x = tf.image.random_contrast(x, 0.8, 1.2)
    
    return x, y

# Apply augmentation to the DATASET, not the model
# num_parallel_calls=tf.data.AUTOTUNE runs this on CPU while GPU trains
train_ds = train_ds.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)

# Optimization
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# ==========================================
# 5. BUILD CLEAN MODEL
# ==========================================
learning_rate = 1e-4
l2_strength = 1e-4    

print("\nBuilding Clean Model...")

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    
    # NOTE: No augmentation layers here!
    # The model receives images that are ALREADY augmented by the dataset.
    
    layers.Rescaling(1./255),
    
    layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
    layers.Dropout(0.5), 
    
    layers.Dense(len(class_names)) 
])

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# ==========================================
# 6. TRAIN
# ==========================================
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "Jan_23_Clean_Best.h5",
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=8, restore_best_weights=True, verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=4, min_lr=1e-6, verbose=1
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stopping, reduce_lr, checkpoint_cb]
)

# ==========================================
# 7. SAVE
# ==========================================
model.save('Jan_23_Threshold.h5') 
print("\n✅ Success! Use 'Jan_23_Clean_Final.h5' for your webcam.")