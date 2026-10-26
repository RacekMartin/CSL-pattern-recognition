import argparse
import csv
import threading
import time
from pathlib import Path

import cv2
import numpy as np
# openCv is unable to display characters like "á"
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
RAW_VIDEOS_DIR = ROOT / "data" / "raw_videos"
CAMERA_COUNT = 3

# Names of the three input cameras
CAMERA_NAMES = ["stred", "lava", "prava"]
# Camera names in display order
DISPLAY_ORDER = ["lava", "stred", "prava"]

# Input resolution and frame rate
WIDTH, HEIGHT, FPS = 1920, 1080, 30
# Resolution of one camera view in the display window
VIEW_W, VIEW_H = 640, 360

# Signs recorded in one round
LETTERS = ["A", "Á", "E", "É", "Ě", "C", "Č", "R", "Ř", "S", "Š"]
NUMBERS = ["5", "6", "7", "8", "9", "10"]
SIGNS = LETTERS + NUMBERS

# Phase durations in seconds
COUNTDOWN_STEP_S = 0.5
SHOW_S = 1.5
REST_S = 1.5

# Instruction panel under the camera views
INSTRUCTION_H = 300
INSTRUCTION_W = 1920
INSTRUCTION_FONT = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 150)
# Text color (RGB) for each kind of phase
INSTRUCTION_COLORS = {
    "wait": (255, 255, 255),
    "countdown": (255, 200, 0),
    "show": (0, 255, 0),
    "rest": (150, 150, 150),
    "done": (0, 255, 0),
}

# Instantiates camera object
def camera_open(index):
    """
    When no argument specified, VideoCapture uses backend API "MSMF" which
    freezes with my 2 external web cameras borrowed from school.
    Dispite being older version, DirectShow is more reliable and accepts "MJPG" format.
    Only MJPG lets the external cameras deliver 1080p at 30 fps over USB 2.0.
    The integrated camera has no MJPG, but works at 30 fps without it.
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

# Reads frames from one camera and stores them until stop flag is set
def camera_record(camera_name, camera, video_path, start_time, stop_event, latest_frames, times_ms):
    video_writer = None

    while not stop_event.is_set():
        read_successfully, frame = camera.read()
        frame_time_ms = (time.perf_counter() - start_time) * 1000

        # Check if frame was read successfully
        if not read_successfully:
            # Wait 5 ms and try again
            time.sleep(0.005)
            continue

        # Triggered only once - instantiates "video_writer" object
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

# Instrcutions for each sign
def sign_text(sign):
    if sign in NUMBERS:
        return f"číslo {sign}"
    return sign

# Countdown 3-2-1, show, rest
def build_phases(sign):
    phases = []

    for number in ["3", "2", "1"]:
        phases.append({"text": number, "duration_s": COUNTDOWN_STEP_S, "kind": "countdown"})
    phases.append({"text": f"UKÁŽ: {sign_text(sign)}", "duration_s": SHOW_S, "kind": "show"})
    phases.append({"text": "Ruky do lona", "duration_s": REST_S, "kind": "rest"})

    return phases

# Black panel with big centered text (PIL can draw diacritics, OpenCV can't)
def make_instruction_panel(text, kind):
    panel = Image.new("RGB", (INSTRUCTION_W, INSTRUCTION_H), (0, 0, 0))
    draw = ImageDraw.Draw(panel)

    # Determine coors of textbox with if placed at [0,0]
    left, top, right, bottom = draw.textbbox((0, 0), text, font=INSTRUCTION_FONT)
    # justify content center
    x = (INSTRUCTION_W - (right - left)) // 2 - left
    y = (INSTRUCTION_H - (bottom - top)) // 2 - top
    # Write text into black box
    draw.text((x, y), text, font=INSTRUCTION_FONT, fill=INSTRUCTION_COLORS[kind])

    return cv2.cvtColor(np.array(panel), cv2.COLOR_RGB2BGR)

# Three mirrored camera views side by side
def build_camera_row(latest_frames):
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

    return np.hstack(views)

# Shows the preview, guides the participant and adds one event per completed sign
def run_session(latest_frames, start_time, events):
    sign_index = 0
    phases = []
    phase_index = 0
    phase_start = 0.0
    pending_event = None

    while True:
        now = time.perf_counter()

        # Move to the next phase when the current one is over
        if phases and now - phase_start >= phases[phase_index]["duration_s"]:
            if phases[phase_index]["kind"] == "show":
                pending_event = {
                    "label": SIGNS[sign_index],
                    "start_ms": (phase_start - start_time) * 1000,
                    "end_ms": (now - start_time) * 1000,
                }
            phase_index += 1
            phase_start = now

            # Sign complete
            if phase_index == len(phases):
                events.append(pending_event)
                pending_event = None
                phases = []
                sign_index += 1

        # Display instructions
        if phases:
            text = phases[phase_index]["text"]
            kind = phases[phase_index]["kind"]
        elif sign_index < len(SIGNS):
            text = f"Ďalej: {sign_text(SIGNS[sign_index])}"
            kind = "wait"
        else:
            text = "Hotovo!"
            kind = "done"

        # Arrange window content: cameras on top, instructions below
        camera_row = build_camera_row(latest_frames)
        elapsed_s = now - start_time
        shown_number = min(sign_index + 1, len(SIGNS))
        status = (f"REC {elapsed_s:5.1f} s   znak {shown_number}/{len(SIGNS)}"
                  f"   SPACE = start   R = znova   Q = koniec")
        cv2.putText(camera_row, status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        instruction_panel = make_instruction_panel(text, kind)
        cv2.imshow("Recording", np.vstack([camera_row, instruction_panel]))

        # Keyboard controls
        key = cv2.waitKey(15) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" ") and not phases and sign_index < len(SIGNS):
            phases = build_phases(SIGNS[sign_index])
            phase_index = 0
            phase_start = time.perf_counter()
            pending_event = None
        elif key == ord("r"):
            if phases:
                # Sign in progress is discarded
                phases = []
                pending_event = None
            elif sign_index > 0:
                # Sign just finished - return and delete it
                sign_index -= 1
                events.pop()

# Saves the time of every frame of every camera into <prefix>_frames.csv`
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

# Saves when each sign was shown into <prefix>_events.csv
def save_events(csv_path, events):
    # utf-8-sig keeps diacritics readable, also in Excel
    csv_file = open(csv_path, "w", newline="", encoding="utf-8-sig")
    csv_writer = csv.writer(csv_file)

    csv_writer.writerow(["label", "start_ms", "end_ms"])

    for event in events:
        csv_writer.writerow([event["label"], f"{event['start_ms']:.1f}", f"{event['end_ms']:.1f}"])

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

        # A gap longer than 1.5 times the expected gap = dropped frame
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
    parser = argparse.ArgumentParser(description="Records 3 cameras at once.")
    parser.add_argument("--participant", required=True, help="participant code, e.g. R01")
    parser.add_argument("--session", type=int, required=True, help="round number, e.g. 1")
    parser.add_argument("--cams", type=int, nargs=3, default=[1, 0, 2],
                        metavar=("STRED", "LAVA", "PRAVA"),
                        help="camera indices in order: center, left, right")

    args = parser.parse_args()

    # Output file names
    file_prefix = f"{args.participant}_kolo{args.session}"
    video_paths = {}
    for name in CAMERA_NAMES:
        video_paths[name] = RAW_VIDEOS_DIR / f"{file_prefix}_{name}.avi"
    times_csv_path = RAW_VIDEOS_DIR / f"{file_prefix}_frames.csv"
    events_csv_path = RAW_VIDEOS_DIR / f"{file_prefix}_events.csv"

    # Make sure you don't overwrite recording with the same name
    all_output_paths = list(video_paths.values()) + [times_csv_path, events_csv_path]
    for path in all_output_paths:
        if path.exists():
            print(f"ERR: File {path.name} already exists")
            return

    # Open all three cameras
    cameras = {}
    for i in range(CAMERA_COUNT):
        name = CAMERA_NAMES[i]
        index = args.cams[i]
        camera = camera_open(index)
        if camera is None:
            print(f"ERR: unable to open camera {name} (index {index})")
            for opened_camera in cameras.values():
                opened_camera.release()
            return
        cameras[name] = camera

    # Data shared between the threads
    frame_times = {}             # Unique for each thread
    for name in CAMERA_NAMES:
        frame_times[name] = []
    latest_frames = {}           # Shared resource by all threads
    events = []                  # Unique for each thread
    stop_event = threading.Event()
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

    # Show the preview until q is pressed
    run_session(latest_frames, start_time, events)

    # Stop threads and release resources
    stop_event.set()
    for thread in threads:
        thread.join()
    for camera in cameras.values():
        camera.release()
    cv2.destroyAllWindows()

    # Save results and print statistics
    save_frame_times(times_csv_path, frame_times)
    save_events(events_csv_path, events)
    print_stats(frame_times, video_paths)
    print(f"Uložené znaky: {len(events)}/{len(SIGNS)}")



if __name__ == "__main__":
    main()
