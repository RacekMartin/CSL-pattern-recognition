"""
For some reason when I try to play with videocontent it displays the video
in slow-motion!!!
Also there are completely different methods for resizing the window and the
frames of video compared to live web cam translation.
"""

import cv2

video_path = r"C:\VUT_fit\code\CSL-pattern-recognition\videos\IMG_5148.MOV"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Connection with camera wasnt established")
    raise SystemExit

cv2.namedWindow("Window_one", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Window_one", 800, 1200)

while True:
    success, frame = cap.read()
    if not success:
        break

    cv2.imshow("Window_one", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()