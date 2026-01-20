import cv2
import numpy as np
import tensorflow as tf

# ==========================================
# 1. SETUP
# ==========================================
print("Loading model...")
model = tf.keras.models.load_model('Jan_20_model.h5')
print("✅ Model loaded!")

class_names = ['object_1', 'object_2', 'unknown'] 
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 80.0 # Only show if confidence is high

# ==========================================
# 2. PREDICTION FUNCTION
# ==========================================
def predict_roi(roi_image):
    roi_resized = cv2.resize(roi_image, IMG_SIZE)
    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
    roi_array = np.expand_dims(roi_rgb, axis=0)
    
    predictions = model.predict(roi_array, verbose=0)
    score = tf.nn.softmax(predictions[0])
    
    class_index = np.argmax(score)
    confidence = 100 * np.max(score)
    label = class_names[class_index]
    
    return label, confidence

# ==========================================
# 3. WEBCAM LOOP
# ==========================================
cap = cv2.VideoCapture(0)

print("\n📷 Webcam started!")
print("🟢 Object 1 = GREEN")
print("🔵 Object 2 = BLUE")
print("🔴 Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # --- STEP A: PRECISE EDGE DETECTION (Canny) ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adjust (30, 100) if edges are missing, or (50, 150) if too much noise
    edges = cv2.Canny(blurred, 30, 100)
    
    # Make lines thicker so contours connect
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- STEP B: PROCESS SHAPES ---
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2000: 
            x, y, w, h = cv2.boundingRect(cnt)
            
            # ROI Extraction
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0: continue

            # --- STEP C: PREDICT ---
            label, confidence = predict_roi(roi)

            # --- STEP D: FILTER & COLOR LOGIC ---
            
            # 1. Ignore if confidence is too low OR if it is 'unknown'
            if confidence < CONFIDENCE_THRESHOLD or label == 'unknown':
                continue

            # 2. Assign Colors
            if label == 'object_1':
                color = (0, 255, 0) # Green (B, G, R)
            elif label == 'object_2':
                color = (255, 0, 0) # Blue (B, G, R)
            else:
                continue # Safety skip

            # --- STEP E: DRAW ---
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            text = f"{label} {confidence:.0f}%"
            cv2.putText(frame, text, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Debug View (Optional: see what the edges look like)
    # cv2.imshow('Debug: Edges', dilated)

    cv2.imshow('Multi-Object Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()