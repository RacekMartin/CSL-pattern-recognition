import argparse
import csv
from pathlib import Path

import time
import numpy as np

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# Directory paths
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
LANDMARKS_DIR = DATA_DIR / "landmarks"
# CSV file paths
RECORDINGS_CSV = DATA_DIR / "recordings.csv"
SEGMENTS_CSV = DATA_DIR / "segments.csv"
# Hands, Face and Pose model paths
MODELS_DIR = ROOT / "models"

# MediaPipe models
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
FACE_MODEL_PATH = MODELS_DIR / "face_landmarker.task"
POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_full.task"
# Of all the models, only pose model has versions, I'm using "full"
POSE_MODEL_NAME = "full"

# Names of the three input cameras
CAMERA_NAMES = ["stred", "lava", "prava"]

# Threshold is set to low so uncertain detections are kept as well
MIN_CONFIDENCE = 0.3

# Constants to make code more readable
LEFT_SLOT = 0
RIGHT_SLOT = 1
HAND_WRIST = 0

# Within pose landmarks, I'm only interested in 11-22 where left_wrist=15 and right_wrist=16
POSE_IDX = list(range(11, 23))
POSE_LEFT_WRIST = POSE_IDX.index(15)
POSE_RIGHT_WRIST = POSE_IDX.index(16)
MIN_WRIST_VISIBILITY = 0.5

# Print progress every 300 frames to know if extraction is going well
PROGRESS_EVERY_FRAMES = 300

# Within face landmarks, I'm only interested in lips and refference points
lip_points = set()
for connection in vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS:
    lip_points.add(connection.start)
    lip_points.add(connection.end)

LIP_IDX = sorted(lip_points)
FACE_REF_IDX = [33, 133, 362, 263, 1, 152] # eye corners, nose tip and chin
FACE_IDX = LIP_IDX + FACE_REF_IDX

# Output CSV files column names
RECORDINGS_HEADER = ["recording", "participant", "input", "type", "camera", "width", "height",
                     "fps", "mirrored", "pose_model", "mediapipe", "note"]
SEGMENTS_HEADER = ["recording", "start", "end", "label", "mode"]

#Creates hand, pose and face detectors
def create_detectors():
    # Options for hands detector
    hand_options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=MIN_CONFIDENCE,
        min_hand_presence_confidence=MIN_CONFIDENCE,
        min_tracking_confidence=MIN_CONFIDENCE
        # result_callback
    )
    # Options for face detector
    face_options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=MIN_CONFIDENCE,
        min_face_presence_confidence=MIN_CONFIDENCE,
        min_tracking_confidence=MIN_CONFIDENCE
        # output_face_blendshapes
        # output_facial_transformation_matrixes
        # result_callback
    )
    # Options for pose detector
    pose_options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=MIN_CONFIDENCE,
        min_pose_presence_confidence=MIN_CONFIDENCE,
        min_tracking_confidence=MIN_CONFIDENCE
        # output_segmentation_masks
    )

    # Create detectors
    hand_detector = vision.HandLandmarker.create_from_options(hand_options)
    face_detector = vision.FaceLandmarker.create_from_options(face_options)
    pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

    return hand_detector, face_detector, pose_detector

# Reads <prefix>_frames.csv
# Returns {camera_name: np.array of frame times in ms}
def load_frame_times(csv_path):
    times_for_each_camera = {}
    for name in CAMERA_NAMES:
        times_for_each_camera[name] = []

    with open(csv_path, "r", newline="", encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            camera = row["camera"]
            # Row i of a camera must belong to frame i of its video
            if int(row["frame"]) != len(times_for_each_camera[camera]):
                raise ValueError(f"{csv_path.name}: unexpected frame number in row {row}")
            times_for_each_camera[camera].append(float(row["t_ms"]))

    for name in CAMERA_NAMES:
        times_for_each_camera[name] = np.array(times_for_each_camera[name], dtype=np.float64)

    return times_for_each_camera

# Reads <prefix>_events.csv
# Returns [{"label": "A", "start_ms": 9280.3, "end_ms": 10791.1}, ...]
def load_events(csv_path):
    # Array of dictionaries for each captured sign
    events = []

    # utf-8-sig: the file starts with an invisible BOM mark craeted by recrding script
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            events.append({
                "label": row["label"],
                "start_ms": float(row["start_ms"]),
                "end_ms": float(row["end_ms"]),
            })

    return events

# Runs all 3 detectors on one frame and returns results
def detect_frame(detectors, frame_bgr, timestamp_ms):
    hand_detector, face_detector, pose_detector = detectors

    # OpenCV produces BGR format while MediaPipe requires RGB wrapped as mp.Image
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    hand_result = hand_detector.detect_for_video(mp_image, timestamp_ms)
    pose_result = pose_detector.detect_for_video(mp_image, timestamp_ms)
    face_result = face_detector.detect_for_video(mp_image, timestamp_ms)

    return hand_result, face_result, pose_result

# Creates two separate arrays from detected pose results
# pose(12, 3) and pose_vis(12, )
def pose_to_arrays(pose_result):
    # Initialize arrays with NaN values
    pose = np.full((len(POSE_IDX), 3), np.nan, dtype=np.float32)
    pose_vis = np.full(len(POSE_IDX), np.nan, dtype=np.float32)

    # Check if any landmarks were detected
    if len(pose_result.pose_landmarks) == 0:
        return pose, pose_vis

    detected_person = pose_result.pose_landmarks[0]
    index = 0
    for landmark_index in POSE_IDX:
        landmark = detected_person[landmark_index]
        pose[index] = [landmark.x, landmark.y, landmark.z]
        pose_vis[index] = landmark.visibility
        index += 1

    return pose, pose_vis

# Creates three separate arrays from detected hand results
# hands(2, 21, 3), hands_world(2, 21, 3) and hands_score (2, )
def hands_to_arrays(hand_result, pose, pose_vis):
    # Initialize arrays with NaN values
    hands = np.full((2, 21, 3), np.nan, dtype=np.float32)
    hands_world = np.full((2, 21, 3), np.nan, dtype=np.float32)
    hands_score = np.full(2, np.nan, dtype=np.float32)

    # Check if any landmarks were detected
    if not hand_result.hand_landmarks:
        return hands, hands_world, hands_score

    # Determine which hand is left and which is right
    slots = assign_slots(hand_result, pose, pose_vis)

    for hand_index, slot in enumerate(slots):
        for point_index, landmark in enumerate(hand_result.hand_landmarks[hand_index]):
            hands[slot, point_index] = [landmark.x, landmark.y, landmark.z]

        for point_index, landmark in enumerate(hand_result.hand_world_landmarks[hand_index]):
            hands_world[slot, point_index] = [landmark.x, landmark.y, landmark.z]

        hands_score[slot] = hand_result.handedness[hand_index][0].score

    return hands, hands_world, hands_score

# Creates one array from detected face results
# face(46, 3) with lips + reference points of eyes, chin and nose
def face_to_array(face_result):
    # Initialize array with NaN values
    face = np.full((len(FACE_IDX), 3), np.nan, dtype=np.float32)

    # Check if any landmarks were detected
    if not face_result.face_landmarks:
        return face

    all_points = face_result.face_landmarks[0]
    for i, point_index in enumerate(FACE_IDX):
        landmark = all_points[point_index]
        face[i] = [landmark.x, landmark.y, landmark.z]

    return face

"""
Decides into which slot each detected hand belongs

This function is called only when pose landmarks is unbale to differentiate 
hands with at lest 50% certainty. Function Uses MediaPipe's left/right labels

Returns one slot per detected hand
"""
def slots_from_labels(hand_result):
    slots = []

    # Recorded videos are not mirrored and MP assumes that the input is mirrored
    for hand_labels in hand_result.handedness:
        best_label = hand_labels[0].category_name
        if best_label == "Left":
            slots.append(RIGHT_SLOT)
        else:
            slots.append(LEFT_SLOT)

    # When two hands are in frame, check if both are detected as left and right
    if (len(slots) == 2) and (slots[0] == slots[1]):
        if slots[0] == LEFT_SLOT:
            free_slot = RIGHT_SLOT
        else:
            free_slot = LEFT_SLOT

        # The less certain hand is probably the wrong one
        score_hand_0 = hand_result.handedness[0][0].score
        score_hand_1 = hand_result.handedness[1][0].score
        if score_hand_0 < score_hand_1:
            slots[0] = free_slot
        else:
            slots[1] = free_slot

    return slots


"""
Decides the slot of every detected hand

This is the primary function while the previous one is used just in case
pose landmarker is unable to classify the hands correctly
"""
def assign_slots(hand_result, pose, pose_vis):
    hand_count = len(hand_result.hand_landmarks)

    # Store landmarks of wrists that can be trusted (handles NaN correctly)
    trusted_wrists = {}
    if pose_vis[POSE_LEFT_WRIST] >= MIN_WRIST_VISIBILITY:
        trusted_wrists[LEFT_SLOT] = pose[POSE_LEFT_WRIST]
    if pose_vis[POSE_RIGHT_WRIST] >= MIN_WRIST_VISIBILITY:
        trusted_wrists[RIGHT_SLOT] = pose[POSE_RIGHT_WRIST]

    # When none can be trusted, invoke the backup function
    if not trusted_wrists:
        return slots_from_labels(hand_result)

    # Find the closest pair (detected hand, trusted pose wrist)
    best_distance = None
    best_hand = None
    best_slot = None
    for hand_index in range(hand_count):
        wrist_landmark = hand_result.hand_landmarks[hand_index][HAND_WRIST]
        for slot, pose_wrist in trusted_wrists.items():
            distance = np.hypot(wrist_landmark.x - pose_wrist[0], wrist_landmark.y - pose_wrist[1])
            if (best_distance is None) or (distance < best_distance):
                best_distance = distance
                best_hand = hand_index
                best_slot = slot

    slots = []
    for hand_index in range(hand_count):
        slots.append(None)

    slots[best_hand] = best_slot
    if hand_count == 2:
        if best_hand == 0:
            other_hand = 1
        else:
            other_hand = 0

        if best_slot == LEFT_SLOT:
            other_slot = RIGHT_SLOT
        else:
            other_slot = LEFT_SLOT

        slots[other_hand] = other_slot

    return slots

# Runs detection on every frame of one video
# Returns a dictionary with all arrays that go into the .npz file
def extract_video(video_path, timestamps_ms):
    frame_count = len(timestamps_ms)

    # One row per frame, initialize arrays with NaN values
    hands = np.full((frame_count, 2, 21, 3), np.nan, dtype=np.float32)
    hands_world = np.full((frame_count, 2, 21, 3), np.nan, dtype=np.float32)
    hands_score = np.full((frame_count, 2), np.nan, dtype=np.float32)
    pose = np.full((frame_count, len(POSE_IDX), 3), np.nan, dtype=np.float32)
    pose_vis = np.full((frame_count, len(POSE_IDX)), np.nan, dtype=np.float32)
    face = np.full((frame_count, len(FACE_IDX), 3), np.nan, dtype=np.float32)

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise IOError(f"Unable to open {video_path.name}")
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # New detectors for every video - VIDEO mode needs timestamps that keep growing
    detectors = create_detectors()
    previous_timestamp_ms = -1
    start_time = time.perf_counter()

    frame_index = 0
    while frame_index < frame_count:
        read_successfully, frame = video.read()
        if not read_successfully:
            break

        # MediaPipe needs milliseconds parameter as progressively increasing integer value
        timestamp_ms = int(round(timestamps_ms[frame_index]))
        if timestamp_ms <= previous_timestamp_ms:
            timestamp_ms = previous_timestamp_ms + 1
        previous_timestamp_ms = timestamp_ms

        hand_result, face_result, pose_result = detect_frame(detectors, frame, timestamp_ms)

        # Handle pose results first to determine handedness
        frame_pose, frame_pose_vis = pose_to_arrays(pose_result)
        frame_hands, frame_hands_world, frame_hands_score = hands_to_arrays(
            hand_result, frame_pose, frame_pose_vis)

        pose[frame_index] = frame_pose
        pose_vis[frame_index] = frame_pose_vis
        hands[frame_index] = frame_hands
        hands_world[frame_index] = frame_hands_world
        hands_score[frame_index] = frame_hands_score
        face[frame_index] = face_to_array(face_result)

        frame_index += 1

        # Print progress every 300 frames
        if frame_index % PROGRESS_EVERY_FRAMES == 0:
            elapsed_s = time.perf_counter() - start_time
            frames_per_s = frame_index / elapsed_s
            remaining_min = (frame_count - frame_index) / frames_per_s / 60
            print(f"    {frame_index:5}/{frame_count} snímok   {frames_per_s:4.1f} snímok/s   "
                  f"zostáva {remaining_min:4.1f} min")

    # The video must have exactly as many frames as frames.csv has rows
    extra_frame_exists, _ = video.read()
    if frame_index < frame_count or extra_frame_exists:
        print(f"WARNING: {video_path.name}: video and frames.csv have different frame counts")

    video.release()
    for detector in detectors:
        detector.close()

    # Real fps = number of gaps between frames / duration in seconds
    duration_s = (timestamps_ms[-1] - timestamps_ms[0]) / 1000
    real_fps = (frame_count - 1) / duration_s

    return {
        "timestamps_ms": timestamps_ms,
        "hands": hands,
        "hands_world": hands_world,
        "hands_score": hands_score,
        "pose": pose,
        "pose_vis": pose_vis,
        "pose_idx": np.array(POSE_IDX, dtype=np.int16),
        "face": face,
        "face_idx": np.array(FACE_IDX, dtype=np.int16),
        "width": width,
        "height": height,
        "fps": real_fps,
    }

# Saves all arrays into one .npz file
# Writes a temporary file first, so an interrupted run never leaves a half-written .npz
def save_npz(npz_path, arrays):
    temporary_path = npz_path.with_name(npz_path.stem + "_tmp.npz")
    np.savez_compressed(temporary_path, **arrays)
    temporary_path.replace(npz_path)

# Converts sign times (ms) to frame indices of one camera
def events_to_segments(events, timestamps_ms):
    segments = []

    for event in events:
        # First frame at or after the start, first frame after the end
        start = int(np.searchsorted(timestamps_ms, event["start_ms"], side="left"))
        end = int(np.searchsorted(timestamps_ms, event["end_ms"], side="right"))

        if end > start:
            segments.append((start, end, event["label"]))
        else:
            print(f"WARNING: sign {event['label']} has no frames, skipping")

    return segments

# Replaces all rows of one recording in csv table while keeping rows of other recordings
def update_csv(csv_path, header, recording, new_rows):
    kept_rows = []

    if csv_path.exists():
        with open(csv_path, "r", newline="", encoding="utf-8-sig") as csv_file:
            csv_reader = csv.reader(csv_file)
            # Reads the first row of the table
            file_header = next(csv_reader, None)

            # Check if the correct csv path is passed as parameter
            if (file_header is not None) and (file_header != header):
                raise ValueError(f"{csv_path.name} has different columns: {file_header}")

            # Read the rest of the records in the csv file
            for row in csv_reader:
                if (len(row) > 0) and (row[0] != recording):
                    kept_rows.append(row)

    # Temporary file first and then replace, so the table is never half-written
    temporary_path = csv_path.with_name(csv_path.stem + "_tmp.csv")
    with open(temporary_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)
        csv_writer.writerows(kept_rows)
        csv_writer.writerows(new_rows)
    temporary_path.replace(csv_path)

# Finds all recorded rounds in raw_videos folder by corresponding events.csv file name
def find_rounds(participant_filter):
    prefixes = []

    # Iterate through all file names, that ends with "_events.csv"
    for events_path in sorted(RAW_VIDEOS_DIR.glob("*_events.csv")):
        prefix = events_path.name.removesuffix("_events.csv") # "R04_kolo3_events.csv" to """R04_kolo3"
        participant = prefix.split("_")[0] # "R04_kolo3_events.csv" to "R04"
        if (participant_filter is None) or (participant == participant_filter):
            prefixes.append(prefix)

    return prefixes

def main():
    parser = argparse.ArgumentParser(description="Extracts MediaPipe landmarks from recorded videos")
    parser.add_argument("--participant", help="only this participant, e.g. R04 (default: all)")
    parser.add_argument("--camera", choices=CAMERA_NAMES, help="only this camera (default: all)")
    parser.add_argument("--overwrite", action="store_true", help="extract again even if the .npz already exists")
    args = parser.parse_args()

    LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)

    if args.camera is None:
        cameras = CAMERA_NAMES
    else:
        cameras = [args.camera]

    prefixes = find_rounds(args.participant)
    if not prefixes:
        print("ERR: no recordings found")
        return

    total_count = len(prefixes) * len(cameras)
    current_number = 0
    run_start_time = time.perf_counter()

    # Iterate through each recording round
    for prefix in prefixes:
        frame_times = load_frame_times(RAW_VIDEOS_DIR / f"{prefix}_frames.csv")
        events = load_events(RAW_VIDEOS_DIR / f"{prefix}_events.csv")
        participant = prefix.split("_")[0]

        # Iterate through each camera
        for camera in cameras:
            current_number += 1
            recording = f"{prefix}_{camera}"
            video_path = RAW_VIDEOS_DIR / f"{recording}.avi"
            npz_path = LANDMARKS_DIR / f"{recording}.npz"
            timestamps_ms = frame_times[camera]

            if not video_path.exists():
                print(f"ERR: {video_path.name} doesn't exist, skipping")
                continue

            # Extract landmarks and store results or skip if already done
            if (npz_path.exists()) and (not args.overwrite):
                print(f"[{current_number}/{total_count}] {recording}: already extracted")
                with np.load(npz_path) as stored:
                    width = int(stored["width"])
                    height = int(stored["height"])
                    fps = float(stored["fps"])
            else:
                print(f"[{current_number}/{total_count}] {recording}: "
                      f"extracting {len(timestamps_ms)} frames")
                arrays = extract_video(video_path, timestamps_ms)
                save_npz(npz_path, arrays)
                width = arrays["width"]
                height = arrays["height"]
                fps = arrays["fps"]

            # Update recordings and segments csv files
            recording_row = [recording, participant, video_path.name, "kolo", camera,
                             width, height, f"{fps:.2f}", 0, POSE_MODEL_NAME, mp.__version__, ""]
            update_csv(RECORDINGS_CSV, RECORDINGS_HEADER, recording, [recording_row])

            segment_rows = []
            for start, end, label in events_to_segments(events, timestamps_ms):
                segment_rows.append([recording, start, end, label, "prompt"])
            update_csv(SEGMENTS_CSV, SEGMENTS_HEADER, recording, segment_rows)

    run_min = (time.perf_counter() - run_start_time) / 60
    print(f"Done in {run_min:.1f} min")



if __name__ == "__main__":
    main()