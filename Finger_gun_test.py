import cv2
import mediapipe as mp
import numpy as np
import pygame
import urllib.request
import os
import time
import math


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

MODEL_PATH = "hand_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/"
    "hand_landmarker.task"
)

# Tune these after trying it
THUMB_ARMED_THRESHOLD = 0.72
THUMB_FIRE_THRESHOLD = 0.52

SHOT_COOLDOWN = 0.20


# --------------------------------------------------
# DOWNLOAD MEDIAPIPE MODEL
# --------------------------------------------------

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmark model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")


# --------------------------------------------------
# SOUND
# --------------------------------------------------

pygame.mixer.pre_init(
    frequency=44100,
    size=-16,
    channels=2,
    buffer=512
)

pygame.init()


def create_pew_sound():
    sample_rate = 44100
    duration = 0.18

    t = np.linspace(
        0,
        duration,
        int(sample_rate * duration),
        endpoint=False
    )

    # Descending laser frequency
    start_freq = 1500
    end_freq = 250

    k = (end_freq - start_freq) / duration

    phase = 2 * np.pi * (
        start_freq * t +
        0.5 * k * t ** 2
    )

    envelope = np.exp(-18 * t)

    wave = (
        np.sin(phase)
        + 0.35 * np.sin(2 * phase)
    )

    wave *= envelope

    audio = np.int16(
        wave / np.max(np.abs(wave)) * 30000
    )

    # Stereo
    stereo = np.column_stack((audio, audio))

    return pygame.sndarray.make_sound(stereo)


pew_sound = create_pew_sound()


# --------------------------------------------------
# GEOMETRY
# --------------------------------------------------

def point(landmarks, index):
    lm = landmarks[index]

    return np.array([
        lm.x,
        lm.y,
        lm.z
    ])


def distance(landmarks, a, b):
    return np.linalg.norm(
        point(landmarks, a) -
        point(landmarks, b)
    )


def angle(landmarks, a, b, c):
    """
    Returns angle ABC in degrees.
    """

    p1 = point(landmarks, a)
    p2 = point(landmarks, b)
    p3 = point(landmarks, c)

    v1 = p1 - p2
    v2 = p3 - p2

    denominator = np.linalg.norm(v1) * np.linalg.norm(v2)

    if denominator == 0:
        return 0

    cos_angle = np.dot(v1, v2) / denominator

    cos_angle = np.clip(
        cos_angle,
        -1.0,
        1.0
    )

    return math.degrees(
        math.acos(cos_angle)
    )


# --------------------------------------------------
# FINGER GUN DETECTION
# --------------------------------------------------

def detect_finger_gun(landmarks):

    # Landmark IDs:
    #
    # Index:  5,6,7,8
    # Middle: 9,10,11,12
    # Ring:   13,14,15,16
    # Pinky:  17,18,19,20

    index_angle = angle(
        landmarks,
        5,
        6,
        8
    )

    middle_angle = angle(
        landmarks,
        9,
        10,
        12
    )

    ring_angle = angle(
        landmarks,
        13,
        14,
        16
    )

    pinky_angle = angle(
        landmarks,
        17,
        18,
        20
    )

    # Index should be straight
    index_extended = index_angle > 155

    # Other fingers should be folded
    middle_curled = middle_angle < 135
    ring_curled = ring_angle < 135
    pinky_curled = pinky_angle < 135

    gun_shape = (
        index_extended
        and middle_curled
        and ring_curled
        and pinky_curled
    )

    # Normalise thumb movement against hand size
    hand_size = distance(
        landmarks,
        0,
        9
    )

    thumb_distance = distance(
        landmarks,
        4,
        5
    )

    if hand_size == 0:
        return False, 0

    thumb_ratio = (
        thumb_distance /
        hand_size
    )

    return gun_shape, thumb_ratio


# --------------------------------------------------
# MEDIAPIPE
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = (
    mp.tasks.vision.HandLandmarker
)

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

RunningMode = (
    mp.tasks.vision.RunningMode
)

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# --------------------------------------------------
# WEBCAM
# --------------------------------------------------

cap = cv2.VideoCapture(0)

armed = False

last_shot = 0

start_time = time.perf_counter()


with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = cap.read()

        if not success:
            break

        # Mirror like a normal webcam
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms = int(
            (time.perf_counter() - start_time)
            * 1000
        )

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        status = "NO GUN"

        thumb_ratio = 0

        if result.hand_landmarks:

            landmarks = result.hand_landmarks[0]

            gun_shape, thumb_ratio = (
                detect_finger_gun(landmarks)
            )

            # -----------------------------
            # DRAW LANDMARKS
            # -----------------------------

            h, w, _ = frame.shape

            for lm in landmarks:

                x = int(lm.x * w)
                y = int(lm.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )

            # -----------------------------
            # STATE MACHINE
            # -----------------------------

            if gun_shape:

                status = "GUN DETECTED"

                # Thumb is raised
                if thumb_ratio > THUMB_ARMED_THRESHOLD:

                    armed = True
                    status = "ARMED"

                # Thumb comes down after being armed
                elif (
                    armed
                    and thumb_ratio
                    < THUMB_FIRE_THRESHOLD
                ):

                    now = time.perf_counter()

                    if (
                        now - last_shot
                        > SHOT_COOLDOWN
                    ):

                        print("PEW!")

                        pew_sound.play()

                        last_shot = now

                        armed = False

            else:
                armed = False

            # -----------------------------
            # MUZZLE FLASH
            # -----------------------------

            if (
                time.perf_counter()
                - last_shot
                < 0.1
            ):

                index_tip = landmarks[8]

                x = int(index_tip.x * w)
                y = int(index_tip.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    30,
                    (0, 255, 255),
                    -1
                )

                cv2.putText(
                    frame,
                    "PEW!",
                    (x + 30, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 255),
                    4
                )

        # -----------------------------
        # DEBUG UI
        # -----------------------------

        cv2.putText(
            frame,
            status,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Thumb: {thumb_ratio:.2f}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Finger Gun ML",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()
pygame.quit()