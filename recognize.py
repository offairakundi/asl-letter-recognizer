import cv2
import mediapipe as mp

# Set up MediaPipe hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Open your Mac's webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        print("Could not access webcam.")
        break

    # Flip image so it feels like a mirror
    frame = cv2.flip(frame, 1)

    # Convert BGR image to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("ASL Hand Tracking", frame)

    # Press Q to close the camera
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
