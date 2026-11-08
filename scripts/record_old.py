import argparse
import csv
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW_VIDEOS_DIR = ROOT / "data" / "raw_videos"
CAMERA_COUNT = 3

# Names of the three input cameras
CAMERA_NAMES = ["stred", "lava", "prava"]
# Camera names in display order
DISPLAY_ORDER = ["lava", "stred", "prava"]

# Input resolutions
WIDTH, HEIGHT, FPS = 1920, 1080, 30
# Resolution of three frames being displayed within on window
VIEW_W, VIEW_H = 640, 360

# Instantiates camera object
def camera_open(index):
    """
    When no argument specified, VideoCapture uses backend API "MSMF" which
    freezes with my 2 external web cameras borrowed from school.
    Dispite being older version, DirectShow is more reliable and accepts "MJPG" format.
    This and "mp4v" are the only formats that all 3 cameras are able to provide with 30fps.
    """
    camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        return None

    # Set camera resolution
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, FPS)
    # Set the four character code representing the frame format
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G"))

    # When everything went successfully, return camera object
    return camera

# Reads frames from one camera and stores them until stop flag is specified
def camera_record(camera_name, camera, video_path, start_time, stop_event, latest_frames, times_ms):
    video_writer = None

    while not stop_event.is_set():
        read_successfully, frame = camera.read()
        frame_time_ms = (time.perf_counter() - start_time) * 1000

        # Check if frame was read successfully
        if not read_successfully:
            # Wait 5ms and try again
            time.sleep(0.005)
            continue

        # This is triggered only once - instantiates "video_writer" object
        if video_writer is None:
            height = frame.shape[0]
            width = frame.shape[1]
            four_char_code = cv2.VideoWriter.fourcc("M", "J", "P", "G")
            video_writer = cv2.VideoWriter(str(video_path), four_char_code, FPS, (width, height), True)

            if not video_writer.isOpened():
                print(f"ERR: Unable to open {video_path.name}")
                break

        video_writer.write(frame)
        times_ms.append(frame_time_ms)
        latest_frames[camera_name] = frame

    if video_writer is not None:
        video_writer.release()

# Displays window with input captured by all 3 cameras
def show_window(latest_frames, start_time, duration_s):
    while True:
        views = []
        for name in DISPLAY_ORDER:
            frame = latest_frames.get(name)
            if frame is None:
                view = np.zeros((VIEW_H, VIEW_W, 3), dtype=np.uint8)
            else:
                view = cv2.resize(frame, (VIEW_W, VIEW_H))
                view = cv2.flip(view, 1)

            cv2.putText(view, name, (10, VIEW_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            views.append(view)

        # Merge the three pictures
        window = np.hstack(views)
        # Calculate and display the time passed
        elapsed_s = time.perf_counter() - start_time
        cv2.putText(window, f"REC {elapsed_s:5.1f} s   q = quit", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        # Display the frames captured
        cv2.imshow("Recording", window)

        # Capture the input from keyboard and decide whether to quit or not
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if (duration_s > 0) and (elapsed_s >= duration_s):
            break

# Saved time of every frame of every camera into <prefix>_frames.csv
def save_frame_times(csv_path, frames_times):
    # Create file descriptor for working with the spreadsheet
    csv_file = open(csv_path, "w", newline="")
    # Instantiate "csv_writer" object
    csv_writer = csv.writer(csv_file)

    csv_writer.writerow(["camera", "frame", "t_ms"])

    for name in CAMERA_NAMES:
        for frame_index, t in enumerate(frames_times[name]):
            csv_writer.writerow([name, frame_index, f"{t:.1f}"])
    csv_file.close()

# AI generated - Prints frame count, real fps, dropped frames and file size for each camera
def print_stats(frame_times, video_paths):
    # At 30 fps, frames should come every 33.3 ms
    expected_gap_ms = 1000 / FPS

    for name in CAMERA_NAMES:
        times_ms = frame_times[name]
        frame_count = len(times_ms)

        # With fewer than 2 frames there are no gaps to measure
        if frame_count < 2:
            print(f"ERR: {name}: didn't capture any frames!")
            continue

        # Gap = time between two frames
        gaps_ms = []
        for i in range(1, frame_count):
            gaps_ms.append(times_ms[i] - times_ms[i - 1])

        # Real fps = number of gaps / total duration in seconds
        duration_s = (times_ms[-1] - times_ms[0]) / 1000
        real_fps = (frame_count - 1) / duration_s

        # A gap longer than 1.5 times the expected gap =  dropped frame
        dropped_frames = 0
        for gap in gaps_ms:
            if gap > 1.5 * expected_gap_ms:
                dropped_frames += 1

        longest_gap_ms = max(gaps_ms)
        size_mb = video_paths[name].stat().st_size / 1_000_000

        print(f"{name:6} {frame_count:5} snímok  {real_fps:5.1f} fps  "
              f"zahodené: {dropped_frames:3}  najdlhšia medzera: {longest_gap_ms:4.0f} ms  "
              f"{size_mb:5.0f} MB")

def main():
    # Process command line arguments
    parser = argparse.ArgumentParser(description="Records 3 cameras at once.")
    parser.add_argument("--participant", required=True, help="participant code, e.g. R01")
    parser.add_argument("--session", type=int, required=True, help="round number, e.g. 1")
    parser.add_argument("--cams", type=int, nargs=3, default=[1,0,2],
                        metavar=("STRED", "LAVA", "PRAVA"), help="camera indices in order: center, left, right")
    parser.add_argument("--seconds", type=float, default=0,
                        help="recording length in seconds, 0 = until q is pressed")

    args = parser.parse_args()

    # Output file names
    file_prefix = f"{args.participant}_kolo{args.session}"
    video_paths = {}
    for name in CAMERA_NAMES:
        video_paths[name] = RAW_VIDEOS_DIR / f"{file_prefix}_{name}.avi"
    times_csv_path = RAW_VIDEOS_DIR / f"{file_prefix}_frames.csv"

    # Make sure you don't overwrite recording with the same name
    all_output_paths = list(video_paths.values()) + [times_csv_path]
    for path in all_output_paths:
        if path.exists():
            print(f"ERR: File with name: {path.name} already exists")
            return

    # Open all three cameras
    cameras = {}
    for i in range(CAMERA_COUNT):
        name = CAMERA_NAMES[i]
        index = args.cams[i]
        camera = camera_open(index)
        if camera is None:
            print(f"ERR: unable to open camera, index: {name}, {index}")
            for opend_camera in cameras.values():
                opend_camera.release()
            return
        cameras[name] = camera

    # Data shared between the threads
    frame_times = {}                    # Unique for each thread
    for name in CAMERA_NAMES:
        frame_times[name] = []
    latest_frames = {}                  # Shared resource by all threads
    stop_event = threading.Event()      # Shared resource by all threads
    start_time = time.perf_counter()

    # Create one recording thread per camera
    threads = []
    for name in CAMERA_NAMES:
        thread = threading.Thread(
            target=camera_record,
            args=(name, cameras[name], video_paths[name], start_time, stop_event,
                  latest_frames, frame_times[name]))
        thread.start()
        threads.append(thread)

    # Show the preview until q is pressed or time runs out
    show_window(latest_frames, start_time, args.seconds)

    # Stop threads and release resources
    stop_event.set()
    for thread in threads:
        thread.join()
    for camera in cameras.values():
        camera.release()
    cv2.destroyAllWindows()

    # Save frame times and print statistics
    save_frame_times(times_csv_path, frame_times)
    print_stats(frame_times, video_paths)



if __name__ == "__main__":
    main()