import time
import cv2

WIDTH, HEIGHT, FPS = 1920, 1080, 30
TEST_SECONDS = 5

# Try indices 0-5, show a frame from each and measure its real fps
for index in range(6):
    camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        print(f"{index}: neexistuje")
        continue

    # Same settings as in record_new.py
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, FPS)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G"))

    read_successfully, frame = camera.read()
    if not read_successfully:
        print(f"{index}: otvorí sa, ale nedáva obraz (virtuálna kamera?)")
        camera.release()
        continue

    # Count how many frames arrive in TEST_SECONDS
    frame_count = 0
    start = time.perf_counter()
    while time.perf_counter() - start < TEST_SECONDS:
        read_successfully, frame = camera.read()
        if read_successfully:
            frame_count += 1
    real_fps = frame_count / TEST_SECONDS

    # Show the last frame for 2 s, so you know which physical camera it is
    cv2.imshow(f"index {index}", cv2.resize(frame, (960, 540)))
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    print(f"{index}: {frame.shape[1]}x{frame.shape[0]}, {real_fps:.1f} fps")
    camera.release()
