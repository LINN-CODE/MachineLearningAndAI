import cv2
import numpy as np
import tensorflow as tf
import time
from threading import Thread

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================

print("Loading model... (This might take a moment)")
try:
    model = tf.keras.models.load_model("Jan_22_model.h5")
    print("✅ Model loaded!")
except OSError:
    print("❌ Error: Model file 'Jan_23_model2.h5' not found.")
    exit()

class_names = ['object_1', 'object_2', 'unknown']
IMG_SIZE = (224, 224)
IP_WEBCAM_URL = "http://172.23.58.243:8080/video"

# ==========================================
# 2. THREADED CAMERA CLASS (Fixes Latency)
# ==========================================
class ThreadedCamera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        # Reduce internal buffer size
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.status = False
        self.frame = None
        self.start()

    def start(self):
        # Read the first frame to ensure we are ready
        ret, first_frame = self.capture.read()
        if ret:
            self.status = True
            self.frame = first_frame
            self.thread.start()
        else:
            print("❌ Error: Could not start video stream.")

    def update(self):
        # Loop endlessly to grab frames and discard old ones
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(0.01) # Slight rest to save CPU

    def get_frame(self):
        return self.status, self.frame

# ==========================================
# 3. INITIALIZE STREAM
# ==========================================

print("Connecting to IP Webcam stream via Threading...")
stream = ThreadedCamera(IP_WEBCAM_URL)
time.sleep(1.0) # Allow camera to warm up

if not stream.status:
    print("❌ Stream failed to open. Check URL/Network.")
    exit()

# ==========================================
# 4. MAIN LOOP (With Frame Skipping)
# ==========================================

prev_time = time.time()
frame_count = 0
FRAME_SKIP = 5  # Analyze 1 out of every 5 frames

# Store previous results to display during skipped frames
cached_label = "Initializing..."
cached_conf = 0.0
cached_color = (100, 100, 100)

while True:
    status, frame = stream.get_frame()
    
    if not status:
        break

    # ---------------------------------------
    # MODEL INFERENCE (Only every Nth frame)
    # ---------------------------------------
    if frame_count % FRAME_SKIP == 0:
        try:
            # Preprocessing
            input_img = cv2.resize(frame, IMG_SIZE)
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            input_img = input_img / 255.0
            input_img = np.expand_dims(input_img, axis=0)

            # Predict
            predictions = model.predict(input_img, verbose=0)
            score = tf.nn.softmax(predictions[0])

            class_index = np.argmax(score)
            cached_conf = float(np.max(score) * 100)
            label = class_names[class_index]

            # Update cached display variables
            if label == "unknown":
                cached_color = (0, 0, 255) # Red
                cached_label = "Unknown"
            else:
                cached_color = (0, 255, 0) # Green
                cached_label = label

        except Exception as e:
            print(f"Prediction Error: {e}")

    frame_count += 1

    # ---------------------------------------
    # DISPLAY (Runs every single frame)
    # ---------------------------------------
    
    # FPS Calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Overlay
    cv2.rectangle(frame, (0, 0), (400, 85), cached_color, -1)
    
    cv2.putText(frame, cached_label, (10, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.putText(frame, f"Conf: {cached_conf:.1f}%", (10, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (250, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Optimized Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("👋 Exited.")