
import cv2
import mediapipe as mp
import csv
import os
import math

DATA_FILE = "asl_dataset.csv"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# ASL letters that use a static handshape
valid_letters = set("ABCDEFGHI KLMNOPQRSTUVWXY".replace(" ", ""))

# Create CSV with a header if it doesn't exist
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["label"] +
            [f"feature_{i}" for i in range(63)]
        )

current_label = None
saved_count = 0

print("Press a letter key to select its ASL handshape.")
print("Press SPACE to save a sample.")
print("Press Q to quit.")

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        print("Could not access webcam.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    landmarks = None

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(
            frame,
            landmarks,
            mp_hands.HAND_CONNECTIONS
        )

    label_text = current_label if current_label else "None"
    cv2.putText(
        frame,
        f"Selected letter: {label_text}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Samples saved: {saved_count}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("ASL Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC key quits the camera
        break

    elif key != 255:
        pressed = chr(key).upper() if key < 128 else ""

        if pressed in valid_letters:
            current_label = pressed
            print(f"Selected letter: {current_label}")

        elif key == 32:  # SPACE
            if current_label is None:
                print("Select a letter first!")
            elif landmarks is None:
                print("No hand detected!")
            else:
                wrist = landmarks.landmark[0]
                middle_mcp = landmarks.landmark[9]

                scale = math.sqrt(
                    (middle_mcp.x - wrist.x) ** 2 +
                    (middle_mcp.y - wrist.y) ** 2
                )

                if scale < 0.0001:
                    print("Hand landmarks too close together. Try again.")
                    continue

                features = []

                for point in landmarks.landmark:
                    x = (point.x - wrist.x) / scale
                    y = (point.y - wrist.y) / scale
                    z = (point.z - wrist.z) / scale

                    features.extend([x, y, z])

                with open(DATA_FILE, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([current_label] + features)

                saved_count += 1
                print(f"Saved {current_label}: sample {saved_count}")

cap.release()
cv2.destroyAllWindows()
hands.close()
