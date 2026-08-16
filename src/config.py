"""Public defaults for vision and continuous J1/J2 ServoJ tracking."""

import os

# Identity recognition hysteresis.
LOCK_THRESHOLD = 0.35
LOST_THRESHOLD = 0.30
TARGET_SWITCH_CONFIRM_FRAMES = 3
FACE_SCORE_THRESHOLD = 0.60
VISION_REACQUIRE_GRACE_FRAMES = 60

# Short partial-face continuity. CSRT is not an identity recognizer, so its
# real-robot authority is deliberately brief and slower than fresh recognition.
ENABLE_PARTIAL_FACE_TRACKER = True
TRACKER_INIT_CONFIRM_FRAMES = 3
TRACKER_REINIT_INTERVAL = 5
TRACKER_MAX_LOST_FRAMES = 5
TRACKER_SPEED_SCALE = 0.30
EDGE_MARGIN_PX = 25
EDGE_RECOVERY_ERROR_FRACTION = 0.35

# Image target and smoothing.
TARGET_CENTER_Y = 300
DEAD_ZONE_X = 20
DEAD_ZONE_Y = 20
CENTER_SMOOTH_ALPHA = 0.35
ERROR_FILTER_WINDOW = 4
MAX_LOST_FRAMES = 60
DATA_TIMEOUT = 0.5
DISPLAY_HOLD_FRAMES = 5
VISION_LOG_HZ = 5.0

# Software-only eye-line deskew. This changes displayed pixels only and never
# commands J6 or another roll joint.
ENABLE_FACE_DESKEW = True
DESKEW_BUFFER_SIZE = 6
DESKEW_SMOOTH_ALPHA = 0.35
DESKEW_DEAD_ZONE_DEG = 0.35
DESKEW_MAX_ANGLE_DEG = 25.0
DESKEW_MAX_STEP_DEG = 2.0

# Continuous ServoJ loop. These are conservative public defaults and must be
# validated on the local robot before any increase.
SERVO_CONTROL_HZ = 20
SERVO_COMMAND_T = 0.10
SERVO_AHEADTIME = 50.0
SERVO_GAIN = 300.0
SERVO_KP_YAW_VEL = 0.035       # deg/s per horizontal pixel
SERVO_KP_PITCH_VEL = 0.030     # deg/s per vertical pixel
SERVO_MAX_YAW_VEL = 6.0        # deg/s
SERVO_MAX_PITCH_VEL = 4.0      # deg/s
SERVO_MAX_YAW_ACCEL = 15.0     # deg/s^2
SERVO_MAX_PITCH_ACCEL = 10.0   # deg/s^2
SERVO_STARTUP_MAX_YAW_VEL = 2.0
SERVO_STARTUP_MAX_PITCH_VEL = 1.5
SERVO_LOG_HZ = 2.0

# Startup and joint-window protection.
STARTUP_HOLD_SECONDS = 1.0
STARTUP_SLOW_SECONDS = 1.5
STARTUP_SPEED_FACTOR = 20
REAL_SPEED_FACTOR = 40
J1_TRACK_WINDOW_DEG = 25.0
J2_TRACK_WINDOW_DEG = 15.0

# This sign is installation-specific. Run tools/calibrate_yaw_direction.py
# before real tracking and replace the value with the measured recommendation.
J1_OUTPUT_SIGN = 1

# DOBOT TCP/IP settings. Public releases do not store a lab network address.
ROBOT_IP = os.getenv("DOBOT_ROBOT_IP", "").strip()
DASHBOARD_PORT = 29999

# Public builds always start in simulation. main_integrate.py changes this only
# after --real-robot and explicit user confirmations.
SIMULATION_MODE = True
