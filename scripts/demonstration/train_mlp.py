import csv
import time
import numpy as np
from pathlib import Path
from collections import Counter

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

from features import frame_features

# Directory paths
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
LANDMARKS_DIR = DATA_DIR / "landmarks"
SEGMENTS_CSV = DATA_DIR / "segments.csv"
MODEL_PATH = ROOT / "models" / "mlp_v1.joblib"

"""
During in person recording, the script was already performing segmentation. 
Due to peoples late reactions I will try to extend the segmentation window
by 15 frames which is ~0.5s and should not overlap the following sign 
"""
SHIFT_FRAMES = 15

# Returns content of segments.csv as a dictionary {"R01_kolo1_lava":  [(267, 312, "A"), ...]}
def load_segments():
    segments_by_recording = {}

    with open(SEGMENTS_CSV, "r", newline="", encoding="utf-8-sig") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            recording = row["recording"]
            if (recording not in segments_by_recording):
                segments_by_recording[recording] = []

            # For example (268, 313, "A")
            segment = (int(row["start"]), int(row["end"]), row["label"])
            segments_by_recording[recording].append(segment)

    return segments_by_recording

# Creates the dataset. Sklearn requires training table with one row per frame
# (only when a hand is visible). Returns 4 lists of the same length:
def build_dataset():
    features_rows = [] #features_rows - 86 numbers per frame (X in sklearn)
    labels = []        #labels        - shown sign per frame (y in sklearn)
    participants = []
    segment_ids = []
    segment_id = 0

    # Iterate through each recording
    for recording, segments in load_segments().items():
        participant = recording.split("_")[0]

        with np.load(LANDMARKS_DIR / f"{recording}.npz") as data:
            hands = data["hands"]
            pose = data["pose"]
            width = int(data["width"])
            height = int(data["height"])
        frame_count = len(hands)

        # Iterate through each segment of the recording
        for start, end, label in segments:
            # Shift the segmentation window and prevent accessing out of range
            shifted_start = min(start + SHIFT_FRAMES, frame_count)
            shifted_end = min(end + SHIFT_FRAMES, frame_count)

            # Iterate through each frame of the segment
            for frame in range(shifted_start, shifted_end):
                # Create input vector from frame's landmarks
                features = frame_features(hands[frame], pose[frame], width, height)
                if features is None:
                    continue

                features_rows.append(features)
                labels.append(label)
                participants.append(participant)
                segment_ids.append(segment_id)

            segment_id += 1

    return features_rows, labels, participants, segment_ids


# Creates new untrained model: scaler + MLP chained together
def create_model():
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(128, 64),
            early_stopping=True,
            max_iter=300,
            random_state=0,
            # By default uses ReLU (instead of sigmoid)
            # ...
        )
    )

# For each participant calculates the share of correctly classified frames
def print_accuracy_per_participant(labels, predictions, participants):
    # Number of frames per one participant
    total = Counter()
    # Number of correctly guessed frames per one participant
    correct = Counter()

    for i in range(len(labels)):
        total[participants[i]] += 1
        if predictions[i] == labels[i]:
            correct[participants[i]] += 1

    for participant in sorted(total):
        print(f"  {participant}: {(correct[participant]/total[participant])*100:.2f}%")

# Calculates the majority vote for one segment and  returns the accuracy of
# correctly_recognized_segments / all segments
def segment_accuracy(labels, predictions, segment_ids):
    """
    Every camera is a separate recording. Recording consists of multiple segments (perfomed signs),
    each consists of ~40 frames. The MLP classifier predicts the correct answer for each frames within one segment.
    """
    # Counter of predicted labels within segment
    votes = {}
    # Actual label (sign performed) in the processed segment
    true_label = {}

    # Iterate through all frames of dataset
    for i in range(len(labels)):
        if (segment_ids[i] not in votes):
            votes[segment_ids[i]] = Counter()
            true_label[segment_ids[i]] = labels[i]
        votes[segment_ids[i]][predictions[i]] += 1

    # Calculate the achieved results
    correct = 0
    for segment_id, counter in votes.items():
        winner, _ = counter.most_common(1)[0]
        if winner == true_label[segment_id]:
            correct += 1

    return correct / len(votes)

# Evaluate the performance of classifier - what signs are being mistaken for which.
# Prints the confusion matrix for my knowledge
def print_confusion_matrix(labels, predictions, label_names):
    """
             predicted: A    Á    R
        actually A:     8    2    0
        actually Á:     3    7    0
        actually R:     0    0   10
    """
    matrix = confusion_matrix(labels, predictions, labels=label_names)

    # Print the header
    header = "     predicted: "
    for name in label_names:
        header += f"{name:>5}"
    print(header)

    # Print the rows
    for row_index in range(len(label_names)):
        line = f"actually {label_names[row_index]:>5}: "
        for count in matrix[row_index]:
            line += f"{count:5d}"
        print(line)

def main():
    start_time = time.perf_counter()

    # Create dataset
    features_rows, labels, participants, segment_ids = build_dataset()
    # Sklearn requires one array of numbers
    sklearn_X = np.array(features_rows)
    label_names = sorted(set(labels))

    # Print summary of th whole dataset
    print(f"frame: {len(labels)}   nb of diffrent signs: {len(label_names)}   people  count: {len(set(participants))}")
    print("Frame count per sign:", dict(sorted(Counter(labels).items())))

    # Leave-one-person-out - every frame is predicted by a model that never saw that person
    print("")
    print("Leave-one-person-out")
    """
    Cross validation is performed 8 times - 8 different people, resulting in 8 models.
    Each model leaves one person out and is trained on the frames of 7 remaining people
    """
    predictions = cross_val_predict(create_model(), sklearn_X, labels, groups=participants,
                                    cv=LeaveOneGroupOut(), n_jobs=-1)

    # Overall results
    correct_frames = 0
    for i in range(len(labels)):
        if predictions[i] == labels[i]:
            correct_frames += 1
    print(f"Accuracy (frames): {(correct_frames / len(labels)) * 100:.2f}%")
    print(f"Accuracy (signs, majority vote): {segment_accuracy(labels, predictions, segment_ids) * 100:.2f}%")

    # Per person results
    print_accuracy_per_participant(labels, predictions, participants)
    print()
    print(classification_report(labels, predictions, labels=label_names, zero_division=0))
    print_confusion_matrix(labels, predictions, label_names)

    # Final model for the live demo is trained on every person from dataset
    print("")
    print("Training the final model")
    model = create_model()
    model.fit(sklearn_X, labels)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Done: {MODEL_PATH}   ({time.perf_counter() - start_time:.0f} s)")



if __name__ == "__main__":
    main()