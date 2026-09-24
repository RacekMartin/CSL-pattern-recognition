import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

import time
import numpy as np

"""
To create a detector object there are 2 things needed - model and options.
"""

# Package paths which include models
FACE_MODEL_PATH = r"C:\VUT_bakalarka\code\CSL-pattern-recognition\models\face_landmarker.task"
HAND_MODEL_PATH = r"C:\VUT_bakalarka\code\CSL-pattern-recognition\models\hand_landmarker.task"

# Visual effects connecting landmarks on video
HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS
LIPS_CONNECTIONS = vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS

# Options for hands detector
hand_options = vision.HandLandmarkerOptions(
    base_options = mp.tasks.BaseOptions(model_asset_path = HAND_MODEL_PATH),
    running_mode = vision.RunningMode.VIDEO,
    num_hands = 2,
    min_hand_detection_confidence = 0.5,
    min_tracking_confidence = 0.5,
    # min_hand_pressence_confidence
    # result_callback
)

# Options for face detector
face_options = vision.FaceLandmarkerOptions(
    base_options = mp.tasks.BaseOptions(model_asset_path = FACE_MODEL_PATH),
    running_mode = vision.RunningMode.VIDEO,
    num_faces = 1,
    min_face_detection_confidence = 0.5,
    min_tracking_confidence = 0.5,
    # min_face_pressence_confidence
    # output_face_blendshapes
    # output_facial_transformation_matrixes
    # result_callback
)

# Create detectores themselves
hand_detector = vision.HandLandmarker.create_from_options(hand_options)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

"""
Performs landmark detection.

Args:
    frame: frame captured from the camera in main loop
    hand_detector: hand detecting object
    face_detector: face detecting object
    timestamp_ms: timestemp of the currently processed frame
Returns:
    (hand_result, face_results): two mp objects as a result of detection
"""
def mp_detection(frame, hand_detector, face_detector, timestamp_ms):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    hand_results = hand_detector.detect_for_video(mp_image, timestamp_ms)
    face_results = face_detector.detect_for_video(mp_image, timestamp_ms)

    """
    These lines of code are temporary. It's just the coordinates of
    detected landmarks so I know something is happening.
    
    print(f"Detected {len(hand_results.hand_landmarks)} hand(s)")
    for hand in hand_results.hand_landmarks:
        for i, landmark in enumerate(hand):
            print(f"point{i}: x={landmark.x}, y={landmark.y}, z={landmark.z}")
    """
    return hand_results, face_results

"""
Visualizes the connections between landmarks.

Args:
    frame: frame captured from the camera in main loop
    hand_results: numpy array of hand landmarks 
    face_results: numpy array of facial landmarks 
Returns:
    none
"""
def draw_landmarks(frame, hand_results, face_results):
    frame_height, frame_width, _ = frame.shape

    # Iterate through each detected hand (each defined by 21 landmarks)
    for hand in hand_results.hand_landmarks:

        # Convert from normalized to pixel coordinates
        points = []
        for landmark in hand:
            x_coor = int(landmark.x * frame_width)
            y_coor = int(landmark.y * frame_height)
            points.append((x_coor, y_coor))

        # Display hands connections
        for connection in HAND_CONNECTIONS:
            start = points[connection.start]
            end = points[connection.end]
            cv2.line(frame, start, end, (0, 255, 0), 2)

        for point in points:
            cv2.circle(frame, point, 4, (0 ,0 ,255), -1)

    # When there's face in the camera, display lips connections
    if face_results.face_landmarks:
        face = face_results.face_landmarks[0]
        points = []
        for landmark in face:
            x_coor = int(landmark.x * frame_width)
            y_coor = int(landmark.y * frame_height)
            points.append((x_coor, y_coor))
        for connection in LIPS_CONNECTIONS:
            start = points[connection.start]
            end = points[connection.end]
            cv2.line(frame, start, end, (255, 0, 0), 1)

# Main loop
web_cam = cv2.VideoCapture(0)

if not web_cam.isOpened():
    print("Camera connection wasn't established")
    raise SystemExit

web_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
web_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # Read camera input
    success, frame = web_cam.read()
    if not success:
        print("Coudn't read a frame from the camera")
        break

    timestamp_ms = int(time.monotonic() * 1000)

    # Detect facial and hands landmarks
    hand_results, face_results = mp_detection(frame, hand_detector, face_detector, timestamp_ms)

    # Create connections between landmarks and display them in window
    draw_landmarks(frame, hand_results, face_results)
    cv2.imshow("Working window", frame)

    # End the loop when "q" key is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Cleanup
web_cam.release()
hand_detector.close()
face_detector.close()
cv2.destroyAllWindows()
