"""Vision adapter for the teammate's RobotController interface.

TrackingResult is intentionally kept unchanged.  D415 3-D data is exposed
through get_last_3d() and printed as JSON so the robot controller remains
backwards compatible.
"""

from __future__ import annotations

from collections import deque
import json
import math
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config
from face_engine import (
    DemoConfig,
    FaceEngine,
    SmoothPoint,
    make_source,
)
from robot_controller import TrackingResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FaceDeskewer:
    """Keep the target's eye line horizontal with image-only rotation.

    YuNet returns right-eye and left-eye centers at indices 4:8 of each face
    record. The input frame is raw on every call, while ``angle_deg`` stores
    the smoothed correction applied to subsequent frames.
    """

    def __init__(self) -> None:
        self.angle_deg = 0.0
        self.angle_buf: deque[float] = deque(maxlen=config.DESKEW_BUFFER_SIZE)

    @staticmethod
    def _line_angle_deg(face: np.ndarray) -> Optional[float]:
        right_x, right_y = float(face[4]), float(face[5])
        left_x, left_y = float(face[6]), float(face[7])
        dx = left_x - right_x
        dy = left_y - right_y
        if math.hypot(dx, dy) < 8.0:
            return None
        angle = math.degrees(math.atan2(dy, dx))
        # An undirected eye line repeats every 180 degrees. Normalize it to a
        # small roll angle and reject extreme/profile landmark geometry.
        angle = (angle + 90.0) % 180.0 - 90.0
        return angle if abs(angle) <= 45.0 else None

    def update(self, face: np.ndarray) -> Optional[float]:
        residual = self._line_angle_deg(face)
        if residual is None:
            return None

        # The landmarks were measured after applying the current correction.
        # Add the residual back to estimate the raw-camera eye angle.
        raw_estimate = self.angle_deg + residual
        while raw_estimate - self.angle_deg > 90.0:
            raw_estimate -= 180.0
        while raw_estimate - self.angle_deg < -90.0:
            raw_estimate += 180.0
        self.angle_buf.append(raw_estimate)

        target_angle = sum(self.angle_buf) / len(self.angle_buf)
        difference = target_angle - self.angle_deg
        if abs(difference) <= config.DESKEW_DEAD_ZONE_DEG:
            return residual

        step = float(
            np.clip(
                config.DESKEW_SMOOTH_ALPHA * difference,
                -config.DESKEW_MAX_STEP_DEG,
                config.DESKEW_MAX_STEP_DEG,
            )
        )
        self.angle_deg = float(
            np.clip(
                self.angle_deg + step,
                -config.DESKEW_MAX_ANGLE_DEG,
                config.DESKEW_MAX_ANGLE_DEG,
            )
        )
        return residual

    @staticmethod
    def _rotation_matrix(width: int, height: int, angle_deg: float) -> np.ndarray:
        return cv2.getRotationMatrix2D(
            (width / 2.0, height / 2.0), angle_deg, scale=1.0
        )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not config.ENABLE_FACE_DESKEW or abs(self.angle_deg) < 0.05:
            return frame
        height, width = frame.shape[:2]
        # OpenCV image coordinates have Y pointing down, so applying the
        # measured eye-line angle (not its negative) removes that slope.
        matrix = self._rotation_matrix(width, height, self.angle_deg)
        return cv2.warpAffine(
            frame,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def reset(self) -> None:
        """Return to the unrotated camera frame before target reacquisition."""
        self.angle_deg = 0.0
        self.angle_buf.clear()

    def to_raw_geometry(
        self,
        center: tuple[int, int],
        box: tuple[int, int, int, int],
        frame_shape: tuple[int, ...],
        applied_angle_deg: float,
    ) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
        """Map leveled detection pixels back to the unrotated depth frame."""
        height, width = frame_shape[:2]
        if abs(applied_angle_deg) < 0.05:
            return center, box

        inverse = self._rotation_matrix(width, height, -applied_angle_deg)
        x1, y1, x2, y2 = box
        points = np.array(
            [center, (x1, y1), (x2, y1), (x2, y2), (x1, y2)],
            dtype=np.float64,
        )
        homogeneous = np.column_stack([points, np.ones(len(points))])
        mapped = homogeneous @ inverse.T
        mapped[:, 0] = np.clip(mapped[:, 0], 0, width - 1)
        mapped[:, 1] = np.clip(mapped[:, 1], 0, height - 1)

        raw_center = tuple(np.rint(mapped[0]).astype(int))
        corners = mapped[1:]
        raw_box = (
            int(np.floor(corners[:, 0].min())),
            int(np.floor(corners[:, 1].min())),
            int(np.ceil(corners[:, 0].max())),
            int(np.ceil(corners[:, 1].max())),
        )
        return raw_center, raw_box


class Vision:
    def __init__(
        self,
        source: str = "webcam",
        camera_index: int = 0,
        width: int = 800,
        height: int = 600,
    ) -> None:
        self.config = DemoConfig(
            target_image=PROJECT_ROOT / "data" / "targets",
            source=source,
            camera_index=camera_index,
            frame_width=width,
            frame_height=height,
            face_score_threshold=config.FACE_SCORE_THRESHOLD,
            # Recognition uses hysteresis: a stricter threshold to lock and a
            # lower threshold to keep an already locked target.
            similarity_threshold=config.LOST_THRESHOLD,
        )
        self.engine = FaceEngine(self.config)
        self.source = make_source(self.config)
        self.smoother = SmoothPoint(alpha=config.CENTER_SMOOTH_ALPHA)
        self.deskewer = FaceDeskewer()
        self.target_name = "target"
        self.target_switch_pending = False
        self.target_confirm_count = 0
        self.locked = False
        self.lost_frame_count = 0
        self.recognition_streak = 0
        self.recognition_frame_count = 0
        self.tracker = None
        self.last_target_similarity = 0.0
        self.last_tracking_source = "lost"
        self.last_console_log = 0.0
        self.last_display_target = None
        self.display_hold_count = 0
        self.last_detections: list[dict[str, object]] = []
        self.last_3d: dict[str, object] = {
            "depth_m": None,
            "point_camera_m": None,
            "point_base_m": None,
        }

    def read(self):
        return self.source.read()

    def set_target(self, target_name: str, target_dir: Path | str) -> int:
        """Hot-load one SFace identity and reset all old tracking state."""
        clean_name = target_name.strip()
        if not clean_name:
            raise ValueError("target name cannot be empty")

        feature_count = self.engine.set_target(Path(target_dir))
        self.target_name = clean_name
        self.target_switch_pending = True
        self.target_confirm_count = 0
        self.locked = False
        self.lost_frame_count = 0
        self.recognition_streak = 0
        self.recognition_frame_count = 0
        self.tracker = None
        self.last_target_similarity = 0.0
        self.last_tracking_source = "switching"
        self.last_display_target = None
        self.display_hold_count = 0
        self.last_detections = []
        self.last_3d = {
            "depth_m": None,
            "point_camera_m": None,
            "point_base_m": None,
        }
        self.smoother.reset()
        self.deskewer.reset()
        print(
            f"[vision] active target changed to {self.target_name!r}; "
            f"SFace references={feature_count}, awaiting "
            f"{config.TARGET_SWITCH_CONFIRM_FRAMES} confirmations"
        )
        return feature_count

    @staticmethod
    def _new_tracker():
        creator = getattr(cv2, "TrackerCSRT_create", None)
        if callable(creator):
            return creator()
        legacy = getattr(cv2, "legacy", None)
        creator = getattr(legacy, "TrackerCSRT_create", None)
        if callable(creator):
            return creator()
        raise RuntimeError("OpenCV CSRT tracker is unavailable; install opencv-contrib-python.")

    def _log_status(self, message: str) -> None:
        """Limit console I/O so rendering and recognition keep their frame rate."""
        now = time.monotonic()
        if now - self.last_console_log >= 1.0 / config.VISION_LOG_HZ:
            print(message)
            self.last_console_log = now

    def _initialize_tracker(
        self,
        frame: np.ndarray,
        box: tuple[int, int, int, int],
    ) -> None:
        if not config.ENABLE_PARTIAL_FACE_TRACKER:
            self.tracker = None
            return
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        if width < 16 or height < 16:
            self.tracker = None
            return
        tracker = self._new_tracker()
        tracker.init(frame, (int(x1), int(y1), int(width), int(height)))
        self.tracker = tracker

    def _update_partial_tracker(
        self,
        frame: np.ndarray,
    ) -> Optional[tuple[tuple[int, int, int, int], tuple[int, int]]]:
        if self.tracker is None:
            return None
        success, tracked_box = self.tracker.update(frame)
        if not success:
            return None

        x, y, width, height = (float(value) for value in tracked_box)
        frame_h, frame_w = frame.shape[:2]
        raw_center = (
            int(round(x + width / 2.0)),
            int(round(y + height / 2.0)),
        )
        x1 = max(0, min(frame_w - 1, int(round(x))))
        y1 = max(0, min(frame_h - 1, int(round(y))))
        x2 = max(0, min(frame_w - 1, int(round(x + width))))
        y2 = max(0, min(frame_h - 1, int(round(y + height))))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        return (x1, y1, x2, y2), raw_center

    @staticmethod
    def _edge_aware_errors(
        center: tuple[int, int],
        box: tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int]:
        error_x = (frame_width // 2) - center[0]
        error_y = config.TARGET_CENTER_Y - center[1]
        x1, y1, x2, y2 = box
        recovery_x = int(frame_width * config.EDGE_RECOVERY_ERROR_FRACTION)
        recovery_y = int(frame_height * config.EDGE_RECOVERY_ERROR_FRACTION)
        if x1 <= config.EDGE_MARGIN_PX:
            error_x = max(error_x, recovery_x)
        elif x2 >= frame_width - 1 - config.EDGE_MARGIN_PX:
            error_x = min(error_x, -recovery_x)
        if y1 <= config.EDGE_MARGIN_PX:
            error_y = max(error_y, recovery_y)
        elif y2 >= frame_height - 1 - config.EDGE_MARGIN_PX:
            error_y = min(error_y, -recovery_y)
        return int(error_x), int(error_y)

    def _tracked_result(
        self,
        frame: np.ndarray,
        box: tuple[int, int, int, int],
        raw_center: tuple[int, int],
    ) -> TrackingResult:
        frame_h, frame_w = frame.shape[:2]
        center = self.smoother.update(raw_center)
        error_x, error_y = self._edge_aware_errors(
            center,
            box,
            frame_w,
            frame_h,
        )
        depth_center = (
            max(0, min(frame_w - 1, center[0])),
            max(0, min(frame_h - 1, center[1])),
        )
        point = self.source.get_point_camera_m(depth_center, box)
        self.last_3d = {
            "depth_m": round(point[2], 4) if point else None,
            "point_camera_m": [round(value, 4) for value in point] if point else None,
            "point_base_m": None,
        }
        self.last_tracking_source = "tracker"
        self.last_display_target = {
            "bbox": box,
            "similarity": self.last_target_similarity,
            "is_target": True,
            "source": "tracker",
        }
        self.display_hold_count = 0
        self.last_detections.append(
            {
                "bbox": box,
                "similarity": self.last_target_similarity,
                "is_target": True,
                "source": "tracker",
            }
        )
        result = TrackingResult(
            locked=True,
            tracking=True,
            target_name=self.target_name,
            similarity=self.last_target_similarity,
            bbox=box,
            center=center,
            error_x=error_x,
            error_y=error_y,
            frame_width=frame_w,
            frame_height=frame_h,
            timestamp=time.time(),
        )
        result.tracking_source = "tracker"
        payload = {
            "locked": True,
            "tracking": True,
            "tracking_source": "tracker",
            "target_name": self.target_name,
            "similarity": round(result.similarity, 4),
            "bbox": [int(value) for value in box],
            "center": [int(value) for value in center],
            "error_x": error_x,
            "error_y": error_y,
            "frame_width": frame_w,
            "frame_height": frame_h,
            "deskew_angle_deg": round(self.deskewer.angle_deg, 3),
            **self.last_3d,
            "timestamp": result.timestamp,
        }
        self._log_status(f"[vision-json] {json.dumps(payload, ensure_ascii=True)}")
        return result

    def update(self, frame) -> TrackingResult:
        """Recognize the active identity and return the teammate data contract."""
        applied_deskew_angle = self.deskewer.angle_deg
        leveled = self.deskewer.apply(frame)
        if leveled is not frame:
            frame[:] = leveled

        h, w = frame.shape[:2]
        threshold = config.LOST_THRESHOLD if self.locked else config.LOCK_THRESHOLD
        faces = self.engine.recognize(frame, threshold)
        self.last_detections = [
            {
                "bbox": tuple(int(value) for value in item.box),
                "similarity": float(item.similarity),
                "is_target": bool(item.is_target),
                "source": "recognition",
            }
            for item in faces
        ]
        target = next((item for item in faces if item.is_target), None)

        if target is None:
            self.recognition_streak = 0
            if self.target_switch_pending:
                self.target_confirm_count = 0
            best_similarity = max(
                (float(item.similarity) for item in faces),
                default=0.0,
            )
            self.lost_frame_count += 1
            if (
                config.ENABLE_PARTIAL_FACE_TRACKER
                and self.tracker is not None
                and self.lost_frame_count <= config.TRACKER_MAX_LOST_FRAMES
            ):
                tracked = self._update_partial_tracker(frame)
                if tracked is not None:
                    box, raw_center = tracked
                    self._log_status(
                        f"[vision] {self.target_name} TRACKED partial-face "
                        f"frame={self.lost_frame_count}/{config.TRACKER_MAX_LOST_FRAMES}"
                    )
                    return self._tracked_result(frame, box, raw_center)

            reacquire_limit = max(
                config.VISION_REACQUIRE_GRACE_FRAMES,
                config.TRACKER_MAX_LOST_FRAMES,
            )
            if self.lost_frame_count > reacquire_limit:
                self.locked = False
                self.smoother.reset()
                self.deskewer.reset()
                self.tracker = None
                self.last_tracking_source = "lost"
            if (
                self.last_display_target is not None
                and self.display_hold_count < config.DISPLAY_HOLD_FRAMES
            ):
                held_detection = dict(self.last_display_target)
                held_detection["source"] = "hold"
                self.last_detections.append(held_detection)
                self.display_hold_count += 1
            self.last_3d = {
                "depth_m": None,
                "point_camera_m": None,
                "point_base_m": None,
            }
            result = TrackingResult(
                locked=False,
                tracking=False,
                target_name=self.target_name,
                similarity=best_similarity,
                bbox=None,
                center=None,
                error_x=0,
                error_y=0,
                frame_width=w,
                frame_height=h,
                timestamp=time.time(),
            )
            self._log_status(
                f"[vision] {self.target_name} LOST best_similarity={best_similarity:.3f} "
                f"reacquire={self.lost_frame_count}/{reacquire_limit}"
            )
            return result

        if self.target_switch_pending:
            self.target_confirm_count += 1
            self.last_target_similarity = float(target.similarity)
            self.last_tracking_source = "confirming"
            if self.target_confirm_count < config.TARGET_SWITCH_CONFIRM_FRAMES:
                for detection in self.last_detections:
                    if detection["is_target"]:
                        detection["source"] = "confirming"
                self.last_3d = {
                    "depth_m": None,
                    "point_camera_m": None,
                    "point_base_m": None,
                }
                result = TrackingResult(
                    locked=False,
                    tracking=False,
                    target_name=self.target_name,
                    similarity=float(target.similarity),
                    bbox=target.box,
                    center=target.center,
                    error_x=0,
                    error_y=0,
                    frame_width=w,
                    frame_height=h,
                    timestamp=time.time(),
                )
                result.tracking_source = "confirming"
                self._log_status(
                    f"[vision] {self.target_name} CONFIRMING "
                    f"{self.target_confirm_count}/{config.TARGET_SWITCH_CONFIRM_FRAMES} "
                    f"similarity={target.similarity:.3f}"
                )
                return result

            self.target_switch_pending = False
            self.target_confirm_count = 0
            self.smoother.reset()
            self._log_status(f"[vision] {self.target_name} target switch confirmed")

        self.locked = True
        self.lost_frame_count = 0
        self.recognition_streak += 1
        self.recognition_frame_count += 1
        self.last_target_similarity = float(target.similarity)
        self.last_tracking_source = "recognition"
        self.last_display_target = {
            "bbox": tuple(int(value) for value in target.box),
            "similarity": float(target.similarity),
            "is_target": True,
            "source": "recognition",
        }
        self.display_hold_count = 0
        if self.recognition_streak >= config.TRACKER_INIT_CONFIRM_FRAMES:
            if (
                self.tracker is None
                or self.recognition_frame_count % config.TRACKER_REINIT_INTERVAL == 0
            ):
                self._initialize_tracker(frame, target.box)
        if config.ENABLE_FACE_DESKEW:
            self.deskewer.update(target.face)
        center = self.smoother.update(target.center)
        # Required convention: frame center x - target center x.
        error_x = (w // 2) - center[0]
        error_y = config.TARGET_CENTER_Y - center[1]

        depth_center, depth_box = self.deskewer.to_raw_geometry(
            target.center,
            target.box,
            frame.shape,
            applied_deskew_angle,
        )
        point = self.source.get_point_camera_m(depth_center, depth_box)
        self.last_3d = {
            "depth_m": round(point[2], 4) if point else None,
            "point_camera_m": [round(value, 4) for value in point] if point else None,
            # Needs hand-eye calibration and the current robot pose.
            "point_base_m": None,
        }

        result = TrackingResult(
            locked=True,
            tracking=True,
            target_name=self.target_name,
            similarity=float(target.similarity),
            bbox=target.box,
            center=center,
            error_x=int(error_x),
            error_y=int(error_y),
            frame_width=w,
            frame_height=h,
            timestamp=time.time(),
        )
        result.tracking_source = "recognition"
        payload = {
            "locked": result.locked,
            "tracking": result.tracking,
            "tracking_source": "recognition",
            "target_name": result.target_name,
            "similarity": round(result.similarity, 4),
            "bbox": [int(value) for value in result.bbox] if result.bbox else None,
            "center": [int(value) for value in result.center] if result.center else None,
            "error_x": result.error_x,
            "error_y": result.error_y,
            "frame_width": result.frame_width,
            "frame_height": result.frame_height,
            "deskew_angle_deg": round(self.deskewer.angle_deg, 3),
            **self.last_3d,
            "timestamp": result.timestamp,
        }
        self._log_status(f"[vision-json] {json.dumps(payload, ensure_ascii=True)}")
        return result

    def get_last_3d(self) -> dict[str, object]:
        return dict(self.last_3d)

    def get_last_detections(self) -> list[dict[str, object]]:
        return [dict(item) for item in self.last_detections]

    def get_tracking_source(self) -> str:
        return self.last_tracking_source

    def release(self) -> None:
        self.source.release()
