
import cv2
import mediapipe as mp
import numpy as np
import joblib
import math
from collections import deque, Counter

# Load the trained Random Forest model
model = joblib.load("asl_model.pkl")

# Set up MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Open webcam
cap = cv2.VideoCapture(0)

# Keep recent predictions to reduce flickering
recent_predictions = deque(maxlen=7)

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        print("Could not access webcam.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    predicted_letter = "No hand detected"

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]

        # Draw hand landmarks
        mp_drawing.draw_landmarks(
            frame,
            landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # Normalise landmarks exactly as in collect_data.py
        wrist = landmarks.landmark[0]
        middle_mcp = landmarks.landmark[9]

        scale = math.sqrt(
            (middle_mcp.x - wrist.x) ** 2 +
            (middle_mcp.y - wrist.y) ** 2
        )

        if scale >= 0.0001:
            features = []

            for point in landmarks.landmark:
                x = (point.x - wrist.x) / scale
                y = (point.y - wrist.y) / scale
                z = (point.z - wrist.z) / scale

                features.extend([x, y, z])

            # Make a prediction
            prediction = model.predict(
                np.array(features).reshape(1, -1)
            )[0]

            # Smooth predictions across recent frames
            recent_predictions.append(prediction)
            predicted_letter = Counter(
                recent_predictions
            ).most_common(1)[0][0]

    else:
        recent_predictions.clear()

    # Display prediction
    cv2.putText(
        frame,
        f"Predicted letter: {predicted_letter}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.imshow("ASL Live Recognition", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
