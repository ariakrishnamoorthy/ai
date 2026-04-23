import cv2
import csv
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


label = "A"
samples_target = 150
output_file = "asl_data.csv"

base_options = mp_python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
samples = []

print(f"Recording '{label}'. SPACE=capture, Q=quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Webcam not found.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image_obj = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image_obj)

    landmarks = None
    if result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]
        landmarks = []
        h, w, _ = frame.shape
        
        for lm in hand_landmarks:
            landmarks.extend([lm.x, lm.y, lm.z])
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    color = (0, 255, 0) if landmarks else (0, 0, 255)
    cv2.putText(frame, f"Label: {label}  {len(samples)}/{samples_target}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, "SPACE=capture  Q=quit", (10, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.imshow("ASL Data Collector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        if landmarks:
            samples.append(landmarks + [label])
            print(f"  Captured {len(samples)}/{samples_target}")
            if len(samples) >= samples_target:
                print("Done!")
                break
        else:
            print("  No hand.")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

with open(output_file, "a", newline="") as f:
    csv.writer(f).writerows(samples)

print(f"Saved {len(samples)} for '{label}'")
