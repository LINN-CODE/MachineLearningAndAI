import cv2
import numpy as np
import tensorflow as tf

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
# Load your trained model
print("Loading model... (This might take a moment)")
model = tf.keras.models.load_model('Jan_20_model.h5')
print("✅ Model loaded!")

# Define your class names EXACTLY as they appeared in training
# alphabetical order: ['object_1', 'object_2', 'unknown']
class_names = ['object_1', 'object_2', 'unknown'] 

# Set the image size (Must match your training size)
IMG_SIZE = (224, 224)

# ==========================================
# 2. WEBCAM LOOP
# ==========================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("\n📷 Webcam started! Press 'q' to quit.")

while True:
    # 1. Capture frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # 2. Preprocessing
    # Resize to 224x224
    input_img = cv2.resize(frame, IMG_SIZE)
    # Convert BGR -> RGB
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    # Add batch dimension (1, 224, 224, 3)
    input_img = np.expand_dims(input_img, axis=0)

    # 3. Prediction
    predictions = model.predict(input_img, verbose=0)
    score = tf.nn.softmax(predictions[0])
    
    class_index = np.argmax(score)
    confidence = 100 * np.max(score)
    predicted_label = class_names[class_index]

    # 4. Smart Display Logic
    # If "Unknown", show RED box. If Object, show GREEN box.
    if predicted_label == 'unknown':
        box_color = (0, 0, 255) # Red
        display_text = "Unknown / No Object"
    else:
        box_color = (0, 255, 0) # Green
        display_text = f"{predicted_label}"

    # Draw the bar
    cv2.rectangle(frame, (0, 0), (400, 60), box_color, -1)
    
    # Text: Class Name
    cv2.putText(frame, display_text, (10, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Text: Confidence Score
    cv2.putText(frame, f"Confidence: {confidence:.1f}%", (10, 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Show the frame
    cv2.imshow('Real-Time Object Detection', frame)

    # Quit logic
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()