import sys
import time
import numpy as np
from pathlib import Path
from collections import deque

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import joblib

# Directory paths
ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "mlp_v1.joblib"

sys.path.append(str(SCRIPTS_DIR))
from features import frame_features
from landmarks_extraction import create_detectors, hands_to_arrays, pose_to_arrays

# Constants to make code more readable
CAMERA_INDEX = 0
LEFT_SLOT = 0
RIGHT_SLOT = 1
MIN_PROBABILITY = 0.5

# Visual effects connecting landmarks on video
HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS

# OpenCV can't display some letters, so this substitution will be used
DISPLAY_NAMES = {"Á": "A'", "É": "E'", "Ě": "E^", "Č": "C^", "Ř": "R^", "Š": "S^"}

# Visualizes the connections between landmarks. The display image is mirrored.
# The connections are drawn from hand(2, 21, 3) instread of "hand_results.hand_landmarks"
def draw_hands(display, hands):
    frame_height, frame_width, _ = display.shape

    # Iterate through each detected hand
    for slot in LEFT_SLOT, RIGHT_SLOT:
        # Skip the hand that is not in frame
        if np.isnan(hands[slot][0][0]):
            continue

        # Convert coors from normalized to pixel coordinates and mirror them
        points = []
        for point in range(21):
            x_coor = int((1 - hands[slot][point][0]) * frame_width)
            y_coor = int(hands[slot][point][1] * frame_height)
            points.append((x_coor, y_coor))

        # Display hands connections
        for connection in HAND_CONNECTIONS:
            start = points[connection.start]
            end = points[connection.end]
            cv2.line(display, start, end, (0, 255, 0), 2)

        # Display hand landmarks
        for point in points:
            cv2.circle(display, point, 4, (0, 0, 255), -1)

# Average of the last few probability lists, one number per sign
"""
The MLP predicts every frame on its own and its guesses jump between frames.
Averaging the last few frames makes the shown result stable

MLP can not comprehend motion - statis signs will have better accuracy
"""
def average_probabilities(recent_probabilities, class_count):
    # Initialize the probability array to zeros
    average = np.zeros(class_count)

    # Iterate through he probability lists of the last frames (A:0.6, Á:0.3, R:0.1)
    for probability in recent_probabilities:
        # Add up those values within the frames
        for i in range(class_count):
            average[i] += probability[i]

    # Calculate the average
    for i in range(class_count):
        average[i] /= len(recent_probabilities)

    return average

def main():
    # Load the trained model
    model = joblib.load(MODEL_PATH)
    # Ordered list of 17 signs
    class_names = model.classes_

    # Initialize detectors - face landmarks are not used yet
    hand_detector, pose_detector, face_detector = create_detectors()
    face_detector.close()

    web_cam = cv2.VideoCapture(CAMERA_INDEX)
    if not web_cam.isOpened():
        print("Camera connection wasn't established")
        raise SystemExit

    # Circular buffer for the last "maxlen" frames
    recent_probabilities = deque(maxlen=10)
    last_printed = None
    previous_timestamp_ms = -1

    # Main loop
    while True:
        success, frame = web_cam.read()
        if not success:
            print("Couldn't read a frame from the camera")
            break

        frame_height, frame_width, _ = frame.shape

        # VIDEO mode needs whole, strictly increasing milliseconds
        timestamp_ms = int(time.monotonic() * 1000)
        if timestamp_ms <= previous_timestamp_ms:
            timestamp_ms = previous_timestamp_ms + 1
        previous_timestamp_ms = timestamp_ms

        # Detection on the NOT mirrored frame, exactly like the training videos
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        hand_result = hand_detector.detect_for_video(mp_image, timestamp_ms)
        pose_result = pose_detector.detect_for_video(mp_image, timestamp_ms)

        # Create separate array out of detected landmarks
        pose, pose_vis = pose_to_arrays(pose_result)
        hands, _, _ = hands_to_arrays(hand_result, pose, pose_vis)
        features = frame_features(hands, pose, frame_width, frame_height)

        # Check if frame features were detected successfully
        if features is None:
            recent_probabilities.clear()
            text = "-"
            shown_label = None
        else:
            # predict_proba takes a list of samples and returns a list of results
            probabilities = model.predict_proba([features])[0]
            recent_probabilities.append(probabilities)
            average = average_probabilities(recent_probabilities, len(class_names))

            best_index = int(np.argmax(average))      # index of the largest number
            best_label = class_names[best_index]
            best_probability = average[best_index]

            if best_probability >= MIN_PROBABILITY:
                shown_label = best_label
                text = f"{DISPLAY_NAMES.get(best_label, best_label)} {best_probability:.0%}"
            else:
                shown_label = None
                text = "?"

        # Print output to terminal only when the recognized sign changes
        if (shown_label is not None) and (shown_label != last_printed):
            print(f"{shown_label}   ({best_probability:.0%})")
        last_printed = shown_label

        # Window: mirrored image, hand points and the result
        view = cv2.flip(frame, 1)
        draw_hands(view, hands)
        cv2.putText(view, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        cv2.imshow("CZJ live demo (q = quit)", view)

        # End the loop when "q" key is pressed
        if cv2.waitKey(1) == ord("q"):
            break

    # Cleanup
    web_cam.release()
    hand_detector.close()
    pose_detector.close()
    cv2.destroyAllWindows()
    recent_probabilities.clear()



if __name__ == "__main__":
    main()
