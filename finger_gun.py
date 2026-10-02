import cv2
import mediapipe as mp
import numpy as np
import pygame
import math
import time
import os
import urllib.request


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "hand_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/"
    "hand_landmarker.task"
)

# Thumb IP joint:
# Straight thumb tip -> large angle
# Flexed thumb tip   -> smaller angle
THUMB_ARMED_ANGLE = 155
THUMB_FIRE_ANGLE = 153

SHOT_COOLDOWN = 0.10

# Finger detection
EXTENDED_ANGLE = 140
CURLED_ANGLE = 150


# ============================================================
# MEDIAPIPE MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    print("Downloading hand model...")
    urllib.request.urlretrieve(
        MODEL_URL,
        MODEL_PATH
    )
    print("Model downloaded.")


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
    min_tracking_confidence=0.5,
)


# ============================================================
# SOUND
# ============================================================

pygame.mixer.pre_init(
    frequency=44100,
    size=-16,
    channels=2,
    buffer=512
)

pygame.init()


def make_pew_sound():

    sample_rate = 44100
    duration = 0.20

    t = np.linspace(
        0,
        duration,
        int(sample_rate * duration),
        endpoint=False
    )

    start_frequency = 1700
    end_frequency = 250

    sweep = (
        start_frequency
        + (
            end_frequency
            - start_frequency
        )
        * (t / duration)
    )

    phase = (
        2
        * np.pi
        * np.cumsum(sweep)
        / sample_rate
    )

    wave = (
        np.sin(phase)
        + 0.4 * np.sin(
            phase * 2
        )
    )

    # Quick fade
    wave *= np.exp(
        -16 * t
    )

    wave /= np.max(
        np.abs(wave)
    )

    audio = np.int16(
        wave * 30000
    )

    stereo = np.column_stack(
        (audio, audio)
    )

    return pygame.sndarray.make_sound(
        stereo
    )


pew_sound = make_pew_sound()


# ============================================================
# GEOMETRY
# ============================================================

def get_point(hand, index):

    landmark = hand[index]

    return np.array([
        landmark.x,
        landmark.y,
        landmark.z
    ])


def angle(hand, a, b, c):
    """
    Returns angle ABC in degrees.
    """

    point_a = get_point(
        hand,
        a
    )

    point_b = get_point(
        hand,
        b
    )

    point_c = get_point(
        hand,
        c
    )

    vector_1 = (
        point_a
        - point_b
    )

    vector_2 = (
        point_c
        - point_b
    )

    denominator = (
        np.linalg.norm(
            vector_1
        )
        *
        np.linalg.norm(
            vector_2
        )
    )

    if denominator == 0:
        return 0

    cosine = (
        np.dot(
            vector_1,
            vector_2
        )
        /
        denominator
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    return math.degrees(
        math.acos(
            cosine
        )
    )


# ============================================================
# FINGER STATE
# ============================================================

def finger_angles(
    hand,
    mcp,
    pip,
    dip,
    tip
):

    pip_angle = angle(
        hand,
        mcp,
        pip,
        dip
    )

    dip_angle = angle(
        hand,
        pip,
        dip,
        tip
    )

    return (
        pip_angle,
        dip_angle
    )


def is_extended(
    hand,
    mcp,
    pip,
    dip,
    tip
):

    pip_angle, dip_angle = (
        finger_angles(
            hand,
            mcp,
            pip,
            dip,
            tip
        )
    )

    return (
        pip_angle
        > EXTENDED_ANGLE

        and

        dip_angle
        > EXTENDED_ANGLE
    )


def is_curled(
    hand,
    mcp,
    pip,
    dip,
    tip
):

    pip_angle, dip_angle = (
        finger_angles(
            hand,
            mcp,
            pip,
            dip,
            tip
        )
    )

    return (
        pip_angle
        < CURLED_ANGLE

        or

        dip_angle
        < CURLED_ANGLE
    )


# ============================================================
# FINGER GUN DETECTION
# ============================================================

def detect_finger_gun(hand):

    # --------------------------------------------------------
    # INDEX
    # --------------------------------------------------------

    index_extended = is_extended(
        hand,
        5,
        6,
        7,
        8
    )

    # --------------------------------------------------------
    # MIDDLE
    # --------------------------------------------------------

    middle_extended = is_extended(
        hand,
        9,
        10,
        11,
        12
    )

    middle_curled = is_curled(
        hand,
        9,
        10,
        11,
        12
    )

    # --------------------------------------------------------
    # RING
    # --------------------------------------------------------

    ring_curled = is_curled(
        hand,
        13,
        14,
        15,
        16
    )

    # --------------------------------------------------------
    # PINKY
    # --------------------------------------------------------

    pinky_curled = is_curled(
        hand,
        17,
        18,
        19,
        20
    )

    # --------------------------------------------------------
    # SINGLE BARREL
    #
    # Index straight
    # Middle curled
    # Ring curled
    # Pinky curled
    # --------------------------------------------------------

    single_barrel = (
        index_extended
        and middle_curled
        and ring_curled
        and pinky_curled
    )

    # --------------------------------------------------------
    # DOUBLE BARREL
    #
    # Index straight
    # Middle straight
    # Ring curled
    # Pinky curled
    # --------------------------------------------------------

    double_barrel = (
        index_extended
        and middle_extended
        and ring_curled
        and pinky_curled
    )

    # --------------------------------------------------------
    # CHOOSE BARREL
    # --------------------------------------------------------

    if double_barrel:

        gun_detected = True

        barrel_tips = [
            8,
            12
        ]

        barrel_name = "DOUBLE"

    elif single_barrel:

        gun_detected = True

        barrel_tips = [
            8
        ]

        barrel_name = "SINGLE"

    else:

        gun_detected = False

        barrel_tips = []

        barrel_name = None

    # --------------------------------------------------------
    # THUMB IP JOINT
    #
    # MediaPipe thumb:
    #
    # 1 = CMC
    # 2 = MCP
    # 3 = IP
    # 4 = TIP
    #
    # We measure the angle at landmark 3.
    #
    # Straight:
    #
    # 2 -------- 3 -------- 4
    #
    # Large angle (~170°)
    #
    # Flexed:
    #
    # 2 -------- 3
    #             \
    #              4
    #
    # Smaller angle
    # --------------------------------------------------------

    thumb_ip_angle = angle(
        hand,
        2,
        3,
        4
    )

    return (
        gun_detected,
        thumb_ip_angle,
        barrel_tips,
        barrel_name
    )


# ============================================================
# DRAW HAND
# ============================================================

HAND_CONNECTIONS = [

    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Pinky
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Palm
    (0, 17)
]


def draw_hand(
    frame,
    hand
):

    height, width, _ = (
        frame.shape
    )

    points = []

    for landmark in hand:

        x = int(
            landmark.x
            * width
        )

        y = int(
            landmark.y
            * height
        )

        points.append(
            (x, y)
        )

    # Skeleton
    for start, end in HAND_CONNECTIONS:

        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 0),
            2
        )

    # Joints
    for point in points:

        cv2.circle(
            frame,
            point,
            5,
            (0, 0, 255),
            -1
        )

    return points


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

start_time = (
    time.perf_counter()
)


# ============================================================
# PER-HAND STATE
# ============================================================

armed = {
    "Left": False,
    "Right": False
}

last_shot = {
    "Left": 0,
    "Right": 0
}

last_barrel_tips = {
    "Left": [],
    "Right": []
}


# ============================================================
# MAIN LOOP
# ============================================================

with HandLandmarker.create_from_options(
    options
) as landmarker:

    while True:

        success, frame = (
            cap.read()
        )

        if not success:

            print(
                "Could not read webcam."
            )

            break

        # Mirror webcam
        frame = cv2.flip(
            frame,
            1
        )

        # OpenCV uses BGR.
        # MediaPipe wants RGB.
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=(
                mp.ImageFormat.SRGB
            ),
            data=rgb_frame
        )

        timestamp = int(
            (
                time.perf_counter()
                - start_time
            )
            * 1000
        )

        result = (
            landmarker.detect_for_video(
                mp_image,
                timestamp
            )
        )

        # ====================================================
        # PROCESS HANDS
        # ====================================================

        if result.hand_landmarks:

            for i, image_hand in enumerate(
                result.hand_landmarks
            ):

                # --------------------------------------------
                # DRAW USING IMAGE LANDMARKS
                # --------------------------------------------

                points = draw_hand(
                    frame,
                    image_hand
                )

                # --------------------------------------------
                # USE WORLD LANDMARKS FOR 3D GESTURE DETECTION
                # --------------------------------------------

                if (
                    result.hand_world_landmarks
                    and
                    i
                    < len(
                        result.hand_world_landmarks
                    )
                ):

                    geometry_hand = (
                        result
                        .hand_world_landmarks[i]
                    )

                else:

                    geometry_hand = (
                        image_hand
                    )

                # --------------------------------------------
                # LEFT / RIGHT
                # --------------------------------------------

                try:

                    detected_hand = (
                        result
                        .handedness[i][0]
                        .category_name
                    )

                    # Swap because we mirror the webcam
                    if detected_hand == "Left":

                        hand_name = (
                            "Right"
                        )

                    elif detected_hand == "Right":

                        hand_name = (
                            "Left"
                        )

                    else:

                        hand_name = (
                            detected_hand
                        )

                except:

                    hand_name = (
                        f"Hand{i}"
                    )

                # --------------------------------------------
                # CREATE STATE IF NEEDED
                # --------------------------------------------

                if hand_name not in armed:

                    armed[
                        hand_name
                    ] = False

                    last_shot[
                        hand_name
                    ] = 0

                    last_barrel_tips[
                        hand_name
                    ] = []

                # --------------------------------------------
                # DETECT FINGER GUN
                # --------------------------------------------

                (
                    gun_detected,
                    thumb_ip_angle,
                    barrel_tips,
                    barrel_name
                ) = detect_finger_gun(
                    geometry_hand
                )

                status = "NO GUN"

                # =================================================
                # VALID FINGER GUN
                # =================================================

                if gun_detected:

                    status = (
                        f"{barrel_name} GUN"
                    )

                    last_barrel_tips[
                        hand_name
                    ] = barrel_tips

                    # ============================================
                    # ARM
                    #
                    # Thumb distal phalanx straight
                    # ============================================

                    if (
                        thumb_ip_angle
                        > THUMB_ARMED_ANGLE
                    ):

                        armed[
                            hand_name
                        ] = True

                        status = (
                            f"{barrel_name} ARMED"
                        )

                    # ============================================
                    # FIRE
                    #
                    # Thumb distal phalanx flexes
                    # ============================================

                    elif (
                        armed[
                            hand_name
                        ]
                        and
                        thumb_ip_angle
                        < THUMB_FIRE_ANGLE
                    ):

                        current_time = (
                            time.perf_counter()
                        )

                        if (
                            current_time
                            - last_shot[
                                hand_name
                            ]
                            > SHOT_COOLDOWN
                        ):

                            print(
                                f"{hand_name} "
                                f"{barrel_name} "
                                f"PEW! "
                                f"Thumb angle: "
                                f"{thumb_ip_angle:.0f}"
                            )

                            pew_sound.play()

                            last_shot[
                                hand_name
                            ] = current_time

                            armed[
                                hand_name
                            ] = False

                            status = "PEW!"

                else:

                    armed[
                        hand_name
                    ] = False

                # =================================================
                # MUZZLE FLASH
                # =================================================

                time_since_shot = (
                    time.perf_counter()
                    -
                    last_shot[
                        hand_name
                    ]
                )

                if (
                    time_since_shot
                    < 0.12
                ):

                    active_tips = (
                        last_barrel_tips[
                            hand_name
                        ]
                    )

                    # Flash on every barrel
                    for barrel_tip in active_tips:

                        tip = points[
                            barrel_tip
                        ]

                        cv2.circle(
                            frame,
                            tip,
                            28,
                            (0, 255, 255),
                            -1
                        )

                    # PEW text
                    if active_tips:

                        first_tip = (
                            points[
                                active_tips[0]
                            ]
                        )

                        cv2.putText(
                            frame,
                            "PEW!",
                            (
                                first_tip[0]
                                + 30,

                                first_tip[1]
                                - 30
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.3,
                            (0, 255, 255),
                            4
                        )

                # =================================================
                # STATUS TEXT
                # =================================================

                wrist = points[0]

                cv2.putText(
                    frame,
                    f"{hand_name}: {status}",
                    (
                        wrist[0] - 80,
                        wrist[1] + 35
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (255, 255, 255),
                    2
                )

                # Show actual thumb joint angle
                cv2.putText(
                    frame,
                    (
                        f"Thumb IP: "
                        f"{thumb_ip_angle:.0f} deg"
                    ),
                    (
                        wrist[0] - 80,
                        wrist[1] + 60
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (255, 255, 255),
                    1
                )

        # ====================================================
        # GENERAL UI
        # ====================================================

        cv2.putText(
            frame,
            "Q = quit",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # ====================================================
        # SHOW CAMERA
        # ====================================================

        cv2.imshow(
            "Finger Gun",
            frame
        )

        if (
            cv2.waitKey(1)
            & 0xFF
            == ord("q")
        ):

            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

pygame.quit()