# ASL Fingerspelling Recognition System

A computer vision-based system that recognises static American Sign Language (ASL) fingerspelling letters using hand-landmark detection and machine learning.

## About the Project

Communication barriers can make everyday interactions difficult for Deaf and hard-of-hearing individuals, particularly when others do not understand sign language. This project explores how computer vision and machine learning can help make communication more accessible.

The system uses a webcam to detect hand gestures and predict the corresponding ASL fingerspelling letter in real time.

It currently recognises **24 static letters (A–I and K–Y)**. The letters J and Z are excluded because they involve movement over time and require temporal recognition rather than a single static handshape.

## How It Works

The recognition pipeline consists of five main stages:

1. **Image Capture:** A webcam captures the user's hand.
2. **Hand Detection:** MediaPipe detects 21 landmarks across the hand.
3. **Feature Extraction:** Each landmark provides three coordinates (x, y, z), producing 63 numerical features.
4. **Classification:** A Random Forest classifier analyses these features and predicts the corresponding letter.
5. **Output:** The predicted letter is displayed through a live webcam interface built using Gradio.

## Technologies Used

- **Python** – Core programming language
- **OpenCV** – Image and webcam processing
- **MediaPipe** – Hand landmark detection
- **NumPy** – Numerical data processing
- **Scikit-learn** – Random Forest classification
- **Gradio** – Interactive web interface
- **Joblib** – Saving and loading the trained model

## Dataset and Model

The dataset contains **792 hand-sign samples** collected from three different participants under varying angles and lighting conditions.

Each sample contains 63 numerical features extracted from MediaPipe's hand landmarks.

A Random Forest classifier was trained using an 80/20 train-test split.

**Initial test accuracy: 98.74%**

This result is based on a random split of the collected samples. Since samples from the same participants may appear in both training and testing sets, further evaluation with entirely unseen participants is needed to assess generalisation.

## Current Limitations

- Only 24 static ASL fingerspelling letters are supported.
- J and Z are not recognised because they require movement.
- Recognition accuracy can vary with hand positioning, lighting and camera angle.
- The current model requires further testing with people outside the original dataset.

## Future Improvements

- Expand the dataset with more participants and varied hand positions.
- Improve prediction stability and recognition of frequently confused letters.
- Introduce temporal modelling to recognise dynamic signs such as J and Z.
- Evaluate performance on entirely unseen participants.
- Explore a more accessible and widely deployable interface.

## Project Status

**Working prototype — ongoing development.**

This project is an exploration of how computer vision and machine learning can contribute to more accessible communication.

## License

This project is licensed under the MIT License.
