import cv2
import mediapipe as mp
import urllib.request
import os
import time


# -----------------------------
# DOWNLOAD HAND MODEL
# -----------------------------

MODEL_PATH = "hand_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/"
    "hand_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print("Downloading hand tracking model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")


# -----------------------------
# MEDIAPIPE SETUP
# -----------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# Connections between the 21 landmarks
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),

    (0, 5), (5, 6), (6, 7), (7, 8),

    (5, 9), (9, 10), (10, 11), (11, 12),

    (9, 13), (13, 14), (14, 15), (15, 16),

    (13, 17), (17, 18), (18, 19), (19, 20),

    (0, 17)
]


# -----------------------------
# WEBCAM
# -----------------------------

cap = cv2.VideoCapture(0)

start_time = time.perf_counter()


with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Could not access camera")
            break

        # Mirror webcam
        frame = cv2.flip(frame, 1)

        # Convert BGR -> RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert to MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(
            (time.perf_counter() - start_time) * 1000
        )

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # -----------------------------
        # DRAW HAND
        # -----------------------------

        if result.hand_landmarks:

            height, width, _ = frame.shape

            for hand in result.hand_landmarks:

                # Convert landmark positions to pixels
                points = []

                for landmark in hand:

                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    points.append((x, y))

                # Draw connections
                for start, end in HAND_CONNECTIONS:

                    cv2.line(
                        frame,
                        points[start],
                        points[end],
                        (0, 255, 0),
                        2
                    )

                # Draw landmark dots
                for x, y in points:

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 0, 255),
                        -1
                    )

        cv2.imshow(
            "Hand Tracking",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()