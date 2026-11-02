
import cv2
import gradio as gr
import mediapipe as mp
import numpy as np
import joblib
import math
from collections import deque, Counter

# Load your trained model
model = joblib.load("asl_model.pkl")

# Set up MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# Store recent predictions to reduce flickering
recent_predictions = deque(maxlen=7)


def recognise_hand(frame):
    if frame is None:
        return None, "Show your hand to the camera."

    # Gradio supplies an RGB image
    image = frame.copy()
    results = hands.process(image)

    prediction_text = "No hand detected"

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]

        # Draw hand landmarks
        mp_drawing.draw_landmarks(
            image,
            landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # Use the exact normalisation from collect_data.py
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

            prediction = model.predict(
                np.array(features).reshape(1, -1)
            )[0]

            recent_predictions.append(prediction)

            stable_prediction = Counter(
                recent_predictions
            ).most_common(1)[0][0]

            prediction_text = f"Predicted letter: {stable_prediction}"

    else:
        recent_predictions.clear()

    return image, prediction_text


# Build the browser interface
with gr.Blocks(title="ASL Letter Recogniser") as demo:
    gr.Markdown("# ASL Letter Recogniser")
    gr.Markdown(
        "Show one of the 24 supported static ASL fingerspelling "
        "handshapes to your camera."
    )

    webcam = gr.Image(
        sources=["webcam"],
        type="numpy",
        streaming=True,
        label="Your camera"
    )

    prediction = gr.Textbox(
        label="Recognition result",
        value="Show your hand to the camera.",
        interactive=False
    )

    webcam.stream(
        fn=recognise_hand,
        inputs=webcam,
        outputs=[webcam, prediction],
        stream_every=0.5
    )

demo.launch()
