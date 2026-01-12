import cv2
import numpy as np
import tensorflow as tf

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
# Load your trained model
print("Loading model... (This might take a moment)")
model = tf.keras.models.load_model('my_model.h5')
print("✅ Model loaded!")

# Define your class names EXACTLY as they appeared in training
# (Usually alphabetical order of your folder names)
class_names = ['object_1', 'object_2'] 

# Set the image size (Must match your training size)
IMG_SIZE = (224, 224)

# ==========================================
# 2. WEBCAM LOOP
# ==========================================
# Open default webcam (Index 0 is usually the built-in laptop cam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("\n📷 Webcam started! Press 'q' to quit.")

while True:
    # 1. Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # 2. Preprocessing
    # We create a copy to feed into the model (so we don't mess up the display)
    input_img = cv2.resize(frame, IMG_SIZE)
    
    # Convert from BGR (OpenCV standard) to RGB (TensorFlow standard)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    
    # Add the "Batch" dimension (Model expects shape: (1, 224, 224, 3))
    input_img = np.expand_dims(input_img, axis=0)

    # 3. Prediction
    predictions = model.predict(input_img, verbose=0)
    
    # Get the index of the highest probability
    score = tf.nn.softmax(predictions[0])
    class_index = np.argmax(score)
    confidence = 100 * np.max(score)
    
    predicted_label = class_names[class_index]

    # 4. Display Result on Screen
    # Draw a green bar at the top
    cv2.rectangle(frame, (0, 0), (300, 60), (0, 255, 0), -1)
    
    # Text: Class Name
    cv2.putText(frame, f"{predicted_label}", (10, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Text: Confidence Score
    cv2.putText(frame, f"{confidence:.1f}%", (10, 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Show the frame
    cv2.imshow('Real-Time Object Detection', frame)

    # Quit logic (Press 'q')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()