import numpy as np
import math

# Out of pose landmarks, only <11 , 22> were extracted
POSE_LEFT_SHOULDER = 0
POSE_RIGHT_SHOULDER = 1
# Same constants as in landmarks_extraction.py
LEFT_SLOT = 0
RIGHT_SLOT = 1
HAND_WRIST = 0

# Creates input vector from hand(2, 21, 3) and pose(12, 3)
# Converts the normalized MP coordinates into pixels
def frame_features(hands, pose, width, height):
    # Check if at least one hand is present in the frame
    left_hand_detected = not np.isnan(hands[LEFT_SLOT][HAND_WRIST][0])
    right_hand_detected = not np.isnan(hands[RIGHT_SLOT][HAND_WRIST][0])
    if (not left_hand_detected) and (not right_hand_detected):
        return None

    # Check if both shoulder are present in the frame
    left_shoulder = pose[POSE_LEFT_SHOULDER]
    right_shoulder = pose[POSE_RIGHT_SHOULDER]
    if (np.isnan(left_shoulder[0]) or np.isnan(right_shoulder[0])):
        return None

    # Convert shoulder coors into pixels
    left_shoulder_x = left_shoulder[0] * width
    left_shoulder_y = left_shoulder[1] * height
    right_shoulder_x = right_shoulder[0] * width
    right_shoulder_y = right_shoulder[1] * height

    # Coordinate origin - middle between shoulders
    center_x = (left_shoulder_x + right_shoulder_x) / 2
    center_y = (left_shoulder_y + right_shoulder_y) / 2
    # Normalisation unit - shoulder width
    shoulder_width = math.sqrt((left_shoulder_x - right_shoulder_x)**2
                                + (left_shoulder_y - right_shoulder_y)**2)
    if shoulder_width < 1:
        return None


    features = []
    for slot in LEFT_SLOT, RIGHT_SLOT:
        # When the hand is out of frame, append 42 zeros (21 landmarks * 2)
        if np.isnan(hands[slot][HAND_WRIST][0]):
            features.extend([0.0] * 42)
            continue

        # Convert wrist coors into pixels
        wrist_x = hands[slot][HAND_WRIST][0] * width
        wrist_y = hands[slot][HAND_WRIST][1] * height

        # Hand shape - calculate the relative position of hand landmarks to the wrist
        for point in range(1, 21):
            point_x = hands[slot][point][0] * width
            point_y = hands[slot][point][1] * height
            features.append((point_x - wrist_x) / shoulder_width)
            features.append((point_y - wrist_y) / shoulder_width)

        # Hand position - calculate the relative position of wrist to the coordinate origin
        features.append((wrist_x - center_x) / shoulder_width)
        features.append((wrist_y - center_y) / shoulder_width)

    # Append which hand is detected in the frame
    features.append(1.0 if left_hand_detected else 0.0)
    features.append(1.0 if right_hand_detected else 0.0)

    return np.array(features, dtype=np.float32)


