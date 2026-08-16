"""
Safe target-face tracking demo for DOBOT Magician E6 projects.

This script is adapted from the previous team's D415 + DOBOT sorting program,
but all robot movement, teaching, calibration and suction code has been removed.

This script DOES NOT control the robot. It only:
1. registers one target person from one or more reference images,
2. opens a webcam or optional RealSense color stream,
3. detects visible faces,
4. compares each face with the target embedding,
5. draws a red box around the target and blue boxes around other faces,
6. prints tracking status and simulated robot commands.

The RobotController class is intentionally simulation-only by default. It is
kept as a clean place to connect the official DOBOT TCP/IP API later.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
TARGET_LABEL = "target"

YUNET_MODEL = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = MODEL_DIR / "face_recognition_sface_2021dec.onnx"

MODEL_URLS = {
    YUNET_MODEL: "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    SFACE_MODEL: "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

RS_WIDTH = 640
RS_HEIGHT = 480
RS_FPS = 30
OUT_WIDTH = 800
OUT_HEIGHT = 600
# Keep the camera's current native orientation; do not rotate frames.
ROTATE_REALSENSE = False
ROTATE_REALSENSE_CLOCKWISE = True
# Scene-line Hough leveling is disabled. The integrated vision adapter levels
# the image from the target's two YuNet eye landmarks instead.
AUTO_FIX_HORIZONTAL = False
BRIGHTNESS_ALPHA = 1.45
BRIGHTNESS_BETA = 22


def auto_fix_horizontal(raw_frame: np.ndarray, previous_angle: float) -> tuple[np.ndarray, float]:
    """Apply a small image-only roll correction; never commands a robot axis."""
    gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 140, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, math.pi / 180, threshold=50, minLineLength=30, maxLineGap=8
    )
    angles: list[float] = []
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            angle = math.degrees(math.atan2(float(y2 - y1), float(x2 - x1)))
            if -35.0 < angle < 35.0:
                angles.append(angle)
    measured = float(np.median(angles)) if angles else previous_angle
    measured = float(np.clip(measured, -10.0, 10.0))
    angle = 0.20 * measured + 0.80 * previous_angle
    if abs(angle) < 0.5:
        return raw_frame, angle
    h, w = raw_frame.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle, 1.0)
    fixed = cv2.warpAffine(
        raw_frame, matrix, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return fixed, angle


def letterbox(image: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float, int, int]:
    """Resize without stretching and pad with black borders.

    Returns the padded image, scale, and left/top padding in output pixels.
    """
    src_h, src_w = image.shape[:2]
    scale = min(width / src_w, height / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, *image.shape[2:]), dtype=image.dtype)
    pad_x = (width - new_w) // 2
    pad_y = (height - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


@dataclass
class DemoConfig:
    target_image: Path
    mode: str = "detect"
    source: str = "realsense"
    camera_index: int = 0
    frame_width: int = OUT_WIDTH
    frame_height: int = OUT_HEIGHT
    face_score_threshold: float = 0.75
    enrollment_face_score_threshold: float = 0.75
    face_nms_threshold: float = 0.30
    face_top_k: int = 5000
    similarity_threshold: float = 0.35
    dead_zone_px: int = 60
    smoothing_alpha: float = 0.35
    print_hz: float = 5.0
    mirror_display: bool = False
    auto_download_models: bool = True


@dataclass
class FaceResult:
    face: np.ndarray
    box: tuple[int, int, int, int]
    center: tuple[int, int]
    score: float
    similarity: float
    is_target: bool = False


class RobotController:
    """Simulation-only control interface for a later DOBOT integration."""

    def __init__(self, simulation: bool = True) -> None:
        self.simulation = simulation
        self.enabled = False

    def enable(self) -> None:
        if self.simulation:
            print("[robot] Simulation mode enabled. No robot command will be sent.")
        else:
            raise RuntimeError(
                "Real robot mode is intentionally not implemented in this demo. "
                "Connect the official DOBOT API here only after vision is stable."
            )
        self.enabled = True

    def stop(self) -> None:
        print("[robot] STOP requested (simulation).")

    def send_tracking_command(self, command: str, error_x: int, error_y: int) -> None:
        if not self.enabled:
            return
        print(f"[robot] simulate command={command} error=({error_x:+d},{error_y:+d})")


class VideoSource:
    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def release(self) -> None:
        pass

    def get_point_camera_m(
        self,
        center: tuple[int, int],
        box: tuple[int, int, int, int],
    ) -> Optional[tuple[float, float, float]]:
        """Return a target point in camera coordinates when depth is available."""
        return None


class WebcamSource(VideoSource):
    def __init__(self, camera_index: int, width: int, height: int) -> None:
        self.cap = cv2.VideoCapture(camera_index)
        self.out_width = width
        self.out_height = height
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.tilt_angle = 0.0
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open webcam index {camera_index}.")

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        if not ok:
            return None
        frame = cv2.convertScaleAbs(frame, alpha=BRIGHTNESS_ALPHA, beta=BRIGHTNESS_BETA)
        if AUTO_FIX_HORIZONTAL:
            frame, self.tilt_angle = auto_fix_horizontal(frame, self.tilt_angle)
        frame, _, _, _ = letterbox(frame, self.out_width, self.out_height)
        return frame

    def release(self) -> None:
        self.cap.release()


class RealSenseColorSource(VideoSource):
    def __init__(self, width: int, height: int) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "pyrealsense2 is not installed. Use --source webcam, or install "
                "Intel RealSense Python support first."
            ) from exc

        self.rs = rs
        # RealSense post-processing filters operate on depth only. They make
        # the exported 3-D point steadier; RGB face jitter is handled by the
        # vision and controller filters.
        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()
        try:
            self.temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.35)
        except RuntimeError as exc:
            print(f"[warning] RealSense temporal filter option unavailable: {exc}")
        self.raw_width = RS_WIDTH
        self.raw_height = RS_HEIGHT
        self.rotate_realsense = ROTATE_REALSENSE
        self.rotate_clockwise = ROTATE_REALSENSE_CLOCKWISE
        self.width = self.raw_height if self.rotate_realsense else self.raw_width
        self.height = self.raw_width if self.rotate_realsense else self.raw_height
        self.out_width = width
        self.out_height = height
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, RS_WIDTH, RS_HEIGHT, rs.format.bgr8, RS_FPS)
        self.config.enable_stream(rs.stream.depth, RS_WIDTH, RS_HEIGHT, rs.format.z16, RS_FPS)
        self.align = rs.align(rs.stream.color)
        self.lock = threading.Lock()
        self.frame_bgr: Optional[np.ndarray] = None
        self.depth_image: Optional[np.ndarray] = None
        self.color_intrinsics = None
        self.depth_scale = 0.001
        self.image_scale = 1.0
        self.pad_x = 0
        self.pad_y = 0
        self.tilt_angle = 0.0
        self.stopped = True

        print("Starting RealSense D415 RGB-D stream...")
        self.profile = self.pipeline.start(self.config)
        for _ in range(10):
            self.pipeline.wait_for_frames()

        device = self.profile.get_device()
        print("RealSense device:", device.get_info(rs.camera_info.name))
        print("RealSense serial:", device.get_info(rs.camera_info.serial_number))
        color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.color_intrinsics = color_profile.get_intrinsics()
        depth_sensor = device.first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())
        print(f"RealSense depth scale: {self.depth_scale} m/unit")

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.stopped = False
        self.thread.start()

    def _update(self) -> None:
        while not self.stopped:
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # Read the raw RGB frame.  The fixed orientation correction must
            # happen exactly once before resize, detection, or roll analysis.
            color_img = np.asanyarray(color_frame.get_data())
            filtered_depth = self.spatial_filter.process(depth_frame)
            filtered_depth = self.temporal_filter.process(filtered_depth)
            depth_image = np.asanyarray(filtered_depth.get_data()).copy()

            # D435I is mounted in portrait orientation in this setup.  Keep
            # the depth image in the same orientation so pixel/depth lookup
            # remains valid after detection.
            if self.rotate_realsense:
                rotation = (
                    cv2.ROTATE_90_CLOCKWISE
                    if self.rotate_clockwise
                    else cv2.ROTATE_90_COUNTERCLOCKWISE
                )
                color_img = cv2.rotate(color_img, rotation)
                depth_image = cv2.rotate(depth_image, rotation)

            color_img = cv2.convertScaleAbs(
                color_img, alpha=BRIGHTNESS_ALPHA, beta=BRIGHTNESS_BETA
            )
            if AUTO_FIX_HORIZONTAL:
                color_img, self.tilt_angle = auto_fix_horizontal(
                    color_img, self.tilt_angle
                )

            # All later processing receives this single, already upright
            # frame. There are no additional rotation calls downstream.
            color_img, image_scale, pad_x, pad_y = letterbox(
                color_img, self.out_width, self.out_height
            )
            with self.lock:
                self.frame_bgr = color_img
                self.depth_image = depth_image
                self.image_scale = image_scale
                self.pad_x = pad_x
                self.pad_y = pad_y

    def read(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.frame_bgr is None:
                return None
            return self.frame_bgr.copy()

    def get_point_camera_m(
        self,
        center: tuple[int, int],
        box: tuple[int, int, int, int],
    ) -> Optional[tuple[float, float, float]]:
        """Estimate a face point in the D415 color-camera coordinate system."""
        with self.lock:
            if self.depth_image is None or self.color_intrinsics is None:
                return None

            cx, cy = center
            x1, y1, x2, y2 = box
            # Undo the letterbox before looking up the rotated depth image.
            x_rot = int(np.clip((cx - self.pad_x) / self.image_scale, 0, self.width - 1))
            y_rot = int(np.clip((cy - self.pad_y) / self.image_scale, 0, self.height - 1))

            # Use a small central face region instead of one noisy depth pixel.
            half_w = max(2, min(12, int((x2 - x1) * 0.08 / self.image_scale)))
            half_h = max(2, min(12, int((y2 - y1) * 0.08 / self.image_scale)))
            roi = self.depth_image[
                max(0, y_rot - half_h): min(self.height, y_rot + half_h + 1),
                max(0, x_rot - half_w): min(self.width, x_rot + half_w + 1),
            ]
            valid = roi[roi > 0]
            if valid.size == 0:
                return None

            depth_m = float(np.median(valid)) * self.depth_scale
            if not 0.10 <= depth_m <= 10.0:
                return None

            if self.rotate_realsense and self.rotate_clockwise:
                x_rs = y_rot
                y_rs = self.raw_height - 1 - x_rot
            elif self.rotate_realsense:
                x_rs = self.raw_width - 1 - y_rot
                y_rs = x_rot
            else:
                x_rs = x_rot
                y_rs = y_rot
            point = self.rs.rs2_deproject_pixel_to_point(
                self.color_intrinsics,
                [
                    float(np.clip(x_rs, 0, self.raw_width - 1)),
                    float(np.clip(y_rs, 0, self.raw_height - 1)),
                ],
                depth_m,
            )
            return float(point[0]), float(point[1]), float(point[2])

    def release(self) -> None:
        self.stopped = True
        time.sleep(0.2)
        try:
            self.pipeline.stop()
        except Exception:
            pass


class FaceEngine:
    def __init__(self, config: DemoConfig) -> None:
        ensure_models(config.auto_download_models)
        self.config = config

        self.detector = cv2.FaceDetectorYN.create(
            str(YUNET_MODEL),
            "",
            (config.frame_width, config.frame_height),
            config.face_score_threshold,
            config.face_nms_threshold,
            config.face_top_k,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(SFACE_MODEL), "")
        self.target_features: list[np.ndarray] = []
        target_path = Path(config.target_image)
        has_reference = target_path.is_file() or (
            target_path.is_dir()
            and any(
                path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
                for path in target_path.iterdir()
            )
        )
        if has_reference:
            self.set_target(target_path)
        else:
            print("[setup] No target face enrolled. Press S in the camera window to enroll one.")

    def _load_target_features(self, image_path: Path) -> list[np.ndarray]:
        if image_path.is_dir():
            image_paths = sorted(
                path
                for path in image_path.iterdir()
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            )
        else:
            image_paths = [image_path]

        if not image_paths:
            raise RuntimeError(f"No reference images found in: {image_path}")

        features: list[np.ndarray] = []
        skipped: list[str] = []
        for path in image_paths:
            image = cv2.imread(str(path))
            if image is None:
                skipped.append(f"{path.name}: unreadable")
                continue

            # Match the live-camera preprocessing without stretching faces.
            image, _, _, _ = letterbox(image, OUT_WIDTH, OUT_HEIGHT)

            faces = self.detect_faces(image)
            if len(faces) != 1:
                skipped.append(f"{path.name}: expected 1 face, found {len(faces)}")
                continue
            features.append(self.extract_feature(image, faces[0].face))

        if not features:
            raise RuntimeError("No valid reference image contains exactly one detectable face.")

        print(f"[setup] Registered {len(features)}/{len(image_paths)} reference images.")
        for message in skipped:
            print(f"[setup] Skipped {message}")
        return features

    def detect_enrollment_faces(self, frame: np.ndarray) -> list[FaceResult]:
        """Detect faces for enrollment using the stricter reference threshold."""
        self.detector.setScoreThreshold(self.config.enrollment_face_score_threshold)
        try:
            return self.detect_faces(frame)
        finally:
            self.detector.setScoreThreshold(self.config.face_score_threshold)

    def set_target(self, image_path: Path) -> int:
        """Atomically replace the active SFace reference feature set."""
        self.detector.setScoreThreshold(self.config.enrollment_face_score_threshold)
        try:
            new_features = self._load_target_features(Path(image_path))
        finally:
            self.detector.setScoreThreshold(self.config.face_score_threshold)

        self.target_features = new_features
        return len(new_features)

    def detect_faces(self, frame: np.ndarray) -> list[FaceResult]:
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        if faces is None:
            return []

        results: list[FaceResult] = []
        for face in faces:
            x, y, bw, bh = face[:4].astype(int)
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w - 1, x + bw)
            y2 = min(h - 1, y + bh)
            if x2 <= x1 or y2 <= y1:
                continue
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            results.append(
                FaceResult(
                    face=face,
                    box=(x1, y1, x2, y2),
                    center=center,
                    score=float(face[-1]),
                    similarity=0.0,
                )
            )
        return results

    def extract_feature(self, frame: np.ndarray, face: np.ndarray) -> np.ndarray:
        aligned = self.recognizer.alignCrop(frame, face)
        feature = self.recognizer.feature(aligned)
        return feature

    def recognize(self, frame: np.ndarray, threshold: float) -> list[FaceResult]:
        faces = self.detect_faces(frame)
        if not self.target_features:
            return faces
        best_index: Optional[int] = None
        best_similarity = -math.inf

        for idx, item in enumerate(faces):
            feature = self.extract_feature(frame, item.face)
            # Compare against each enrollment image and keep the closest pose.
            # This avoids lowering the score by averaging very different poses.
            item.similarity = max(
                float(
                    self.recognizer.match(
                        target_feature,
                        feature,
                        cv2.FaceRecognizerSF_FR_COSINE,
                    )
                )
                for target_feature in self.target_features
            )
            if item.similarity > best_similarity:
                best_similarity = item.similarity
                best_index = idx

        if best_index is not None and best_similarity >= threshold:
            faces[best_index].is_target = True
        return faces


class SmoothPoint:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.value: Optional[np.ndarray] = None

    def update(self, point: tuple[int, int]) -> tuple[int, int]:
        current = np.array(point, dtype=np.float32)
        if self.value is None:
            self.value = current
        else:
            self.value = self.alpha * current + (1.0 - self.alpha) * self.value
        return int(self.value[0]), int(self.value[1])

    def reset(self) -> None:
        self.value = None


def ensure_models(auto_download: bool) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    missing = [path for path in MODEL_URLS if not path.exists()]
    if not missing:
        return

    if not auto_download:
        lines = ["Missing OpenCV Zoo model files:"]
        lines += [f"- {path.name}: {MODEL_URLS[path]}" for path in missing]
        raise RuntimeError("\n".join(lines))

    for path in missing:
        url = MODEL_URLS[path]
        print(f"[setup] Downloading {path.name}...")
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as exc:
            raise RuntimeError(
                f"Could not download {path.name}.\n"
                f"Download it manually from:\n{url}\n"
                f"and place it in:\n{MODEL_DIR}"
            ) from exc


def choose_command(error_x: int, error_y: int, dead_zone: int) -> str:
    if abs(error_x) <= dead_zone and abs(error_y) <= dead_zone:
        return "HOLD"
    if abs(error_x) >= abs(error_y):
        return "TURN_LEFT" if error_x > 0 else "TURN_RIGHT"
    return "MOVE_UP" if error_y > 0 else "MOVE_DOWN"


def draw_overlay(
    frame: np.ndarray,
    faces: Iterable[FaceResult],
    target: Optional[FaceResult],
    smoothed_center: Optional[tuple[int, int]],
    command: str,
    error_x: int,
    error_y: int,
    dead_zone: int,
) -> None:
    h, w = frame.shape[:2]
    frame_center = (w // 2, h // 2)
    cv2.circle(frame, frame_center, dead_zone, (80, 80, 80), 1)
    cv2.drawMarker(frame, frame_center, (255, 255, 255), cv2.MARKER_CROSS, 18, 1)

    for item in faces:
        x1, y1, x2, y2 = item.box
        color = (0, 0, 255) if item.is_target else (255, 0, 0)
        label = TARGET_LABEL if item.is_target else "OTHER"
        label += f" {item.similarity:.3f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    status = f"{TARGET_LABEL} LOCKED" if target else f"{TARGET_LABEL} LOST"
    status_color = (0, 0, 255) if target else (180, 180, 180)
    cv2.putText(frame, status, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
    cv2.putText(
        frame,
        f"cmd={command} err=({error_x:+d},{error_y:+d})",
        (24, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )

    if target and smoothed_center:
        cv2.circle(frame, smoothed_center, 5, (0, 255, 255), -1)
        cv2.line(frame, frame_center, smoothed_center, (0, 255, 255), 2)


def print_status(
    last_print: float,
    print_hz: float,
    target: Optional[FaceResult],
    command: str,
    error_x: int,
    error_y: int,
    point_camera_m: Optional[tuple[float, float, float]],
    tracking_result: dict[str, object],
) -> float:
    now = time.time()
    if now - last_print < 1.0 / max(print_hz, 0.1):
        return last_print

    if target:
        print(
            f"[vision] {TARGET_LABEL} LOCKED "
            f"similarity={target.similarity:.3f} "
            f"center={target.center} error=({error_x:+d},{error_y:+d}) "
            f"camera_xyz={point_camera_m} command={command}"
        )
    else:
        print(f"[vision] {TARGET_LABEL} LOST command=HOLD")
    print(f"[vision-json] {json.dumps(tracking_result, ensure_ascii=True)}")
    return now


def make_source(config: DemoConfig) -> VideoSource:
    if config.source == "webcam":
        return WebcamSource(config.camera_index, config.frame_width, config.frame_height)
    if config.source == "realsense":
        return RealSenseColorSource(config.frame_width, config.frame_height)
    raise ValueError(f"Unknown source: {config.source}")


def parse_args() -> DemoConfig:
    parser = argparse.ArgumentParser(description="Safe target-face tracking vision demo.")
    parser.add_argument(
        "--mode",
        choices=["detect"],
        default="detect",
        help="Only safe vision-only detect mode is available in this demo.",
    )
    parser.add_argument(
        "--target",
        default="data/targets",
        help="Reference image or folder of reference images for one target person.",
    )
    parser.add_argument("--source", choices=["webcam", "realsense"], default="realsense")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=OUT_WIDTH)
    parser.add_argument("--height", type=int, default=OUT_HEIGHT)
    parser.add_argument("--similarity-threshold", type=float, default=0.35)
    parser.add_argument("--face-score-threshold", type=float, default=0.75)
    parser.add_argument("--dead-zone", type=int, default=60)
    parser.add_argument("--smoothing-alpha", type=float, default=0.35)
    parser.add_argument("--print-hz", type=float, default=5.0)
    parser.add_argument("--mirror", action="store_true", help="Mirror display like a selfie camera.")
    parser.add_argument("--no-auto-download-models", action="store_true")
    args = parser.parse_args()

    target_path = Path(args.target)
    if not target_path.is_absolute():
        target_path = PROJECT_ROOT / target_path

    return DemoConfig(
        target_image=target_path,
        mode=args.mode,
        source=args.source,
        camera_index=args.camera_index,
        frame_width=args.width,
        frame_height=args.height,
        similarity_threshold=args.similarity_threshold,
        face_score_threshold=args.face_score_threshold,
        dead_zone_px=args.dead_zone,
        smoothing_alpha=args.smoothing_alpha,
        print_hz=args.print_hz,
        mirror_display=args.mirror,
        auto_download_models=not args.no_auto_download_models,
    )


def main() -> int:
    config = parse_args()
    print(f"[info] mode={config.mode} source={config.source}")
    robot = RobotController(simulation=True)
    robot.enable()

    try:
        engine = FaceEngine(config)
        source = make_source(config)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    smoother = SmoothPoint(config.smoothing_alpha)
    last_print = 0.0
    window_name = "Safe Target Face Tracking Demo - Q to quit"
    print("[info] Press Q in the video window to quit.")

    empty_frames = 0
    try:
        while True:
            frame = source.read()
            if frame is None:
                empty_frames += 1
                if empty_frames == 1:
                    print("[warning] Waiting for a RealSense frame...")
                if empty_frames >= 100:
                    raise RuntimeError(
                        "RealSense did not provide a color/depth frame. "
                        "Close other camera programs and reconnect the camera."
                    )
                time.sleep(0.05)
                continue
            empty_frames = 0

            display = cv2.flip(frame, 1) if config.mirror_display else frame.copy()
            faces = engine.recognize(display, config.similarity_threshold)
            target = next((item for item in faces if item.is_target), None)

            h, w = display.shape[:2]
            frame_center = (w // 2, h // 2)
            error_x = 0
            error_y = 0
            command = "HOLD"
            smoothed_center: Optional[tuple[int, int]] = None
            point_camera_m: Optional[tuple[float, float, float]] = None

            if target is not None:
                smoothed_center = smoother.update(target.center)
                # Use the same contract as vision.py and RobotController:
                # image target center minus detected face center.
                error_x = frame_center[0] - smoothed_center[0]
                error_y = frame_center[1] - smoothed_center[1]
                command = choose_command(error_x, error_y, config.dead_zone_px)
                point_camera_m = source.get_point_camera_m(target.center, target.box)
            else:
                smoother.reset()
                robot.stop()

            tracking_result = {
                "locked": target is not None,
                "tracking": target is not None,
                "target_name": TARGET_LABEL if target is not None else None,
                "similarity": round(float(target.similarity), 4) if target is not None else None,
                "bbox": [int(value) for value in target.box] if target is not None else None,
                "center": [int(value) for value in smoothed_center] if smoothed_center is not None else None,
                "error_x": int(error_x),
                "error_y": int(error_y),
                "frame_width": int(w),
                "frame_height": int(h),
                "depth_m": round(point_camera_m[2], 4) if point_camera_m else None,
                "point_camera_m": (
                    [float(round(float(value), 4)) for value in point_camera_m]
                    if point_camera_m
                    else None
                ),
                # Filled only after eye-in-hand calibration and robot-pose input.
                "point_base_m": None,
                "timestamp": time.time(),
            }

            draw_overlay(
                display,
                faces,
                target,
                smoothed_center,
                command,
                error_x,
                error_y,
                config.dead_zone_px,
            )
            last_print = print_status(
                last_print,
                config.print_hz,
                target,
                command,
                error_x,
                error_y,
                point_camera_m,
                tracking_result,
            )
            robot.send_tracking_command(command, error_x, error_y)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        source.release()
        cv2.destroyAllWindows()
        robot.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
