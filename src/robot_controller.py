"""Continuous ServoJ controller for target-face camera tracking."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import re
import threading
import time
from typing import Optional, Tuple

import config

try:
    from vendor.dobot_api import DobotApiDashboard
    DOBOT_IMPORT_ERROR = None
except ImportError as exc:
    DobotApiDashboard = None
    DOBOT_IMPORT_ERROR = exc


@dataclass
class TrackingResult:
    locked: bool
    tracking: bool
    target_name: str
    similarity: float
    bbox: Optional[Tuple[int, int, int, int]]
    center: Optional[Tuple[int, int]]
    error_x: int
    error_y: int
    frame_width: int
    frame_height: int
    timestamp: float


class RobotController:
    """Receive vision at camera rate and drive J1/J2 from a 20 Hz servo loop."""

    def __init__(self) -> None:
        self.dashboard = None
        self.simulation = config.SIMULATION_MODE
        self.sim_angles = [0.0] * 6

        self.startup_initial_angles: Optional[list[float]] = None
        self.command_angles: Optional[list[float]] = None
        self.actual_angles: Optional[list[float]] = None
        self.startup_hold_until: Optional[float] = None
        self.startup_slow_until: Optional[float] = None
        self.normal_speed_restored = False

        self.err_x_buf = deque(maxlen=config.ERROR_FILTER_WINDOW)
        self.err_y_buf = deque(maxlen=config.ERROR_FILTER_WINDOW)
        self.latest_err_x = 0.0
        self.latest_err_y = 0.0
        self.latest_similarity = 0.0
        self.latest_source = "lost"
        self.latest_timestamp = 0.0
        self.target_valid = False
        self.lost_frame_count = 0

        self.yaw_velocity = 0.0
        self.pitch_velocity = 0.0
        self.last_vision_log = 0.0
        self.last_servo_log = 0.0
        self.last_actual_read = 0.0

        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._servo_thread: Optional[threading.Thread] = None
        self._worker_error: Optional[str] = None

    @staticmethod
    def _response_code(response) -> Optional[int]:
        match = re.match(r"\s*(-?\d+)\s*,", str(response))
        return int(match.group(1)) if match else None

    @staticmethod
    def _parse_joint_angles(response) -> Optional[list[float]]:
        groups = re.findall(r"\{([^{}]*)\}", str(response))
        numeric_text = groups[-1] if groups else str(response)
        values = re.findall(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
            numeric_text,
        )
        try:
            angles = [float(value) for value in values]
        except ValueError:
            return None
        return angles[:6] if len(angles) >= 6 else None

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(value, upper))

    @classmethod
    def _approach(cls, current: float, desired: float, max_change: float) -> float:
        return current + cls._clamp(desired - current, -max_change, max_change)

    def _read_current_angles(self) -> list[float]:
        if self.simulation:
            return list(self.sim_angles)
        response = self.dashboard.GetAngle()
        angles = self._parse_joint_angles(response)
        if angles is None:
            raise RuntimeError(f"无法解析机械臂角度：{response!r}")
        return angles

    def connect(self) -> None:
        """Connect and capture the six-joint pose; do not move yet."""
        if self.simulation:
            self.startup_initial_angles = list(self.sim_angles)
            self.command_angles = list(self.sim_angles)
            self.actual_angles = list(self.sim_angles)
            print("[robot] ServoJ simulation enabled; no robot command will be sent")
            self._start_servo_loop()
            return

        if DobotApiDashboard is None:
            raise RuntimeError(f"DOBOT Python SDK import failed: {DOBOT_IMPORT_ERROR}")
        if not config.ROBOT_IP:
            raise RuntimeError(
                "DOBOT_ROBOT_IP is not set. Configure it in the current shell "
                "before requesting real-robot mode."
            )

        self.dashboard = DobotApiDashboard(config.ROBOT_IP, config.DASHBOARD_PORT)
        angles = self._read_current_angles()
        self.startup_initial_angles = list(angles)
        self.command_angles = list(angles)
        self.actual_angles = list(angles)
        print(
            "[robot] connected; captured servo hold pose "
            f"J1={angles[0]:.2f} deg J2={angles[1]:.2f} deg"
        )

    def enable_real_robot(self) -> bool:
        if self.simulation:
            return True
        print("\n===== REAL SERVOJ MODE WARNING =====")
        print("The arm will receive continuous J1/J2 targets at 20 Hz.")
        print("Keep the emergency stop reachable and clear the full workspace.")
        if input("Type E to enable continuous tracking: ").strip().upper() != "E":
            print("[safety] cancelled; no ServoJ loop started")
            return False

        enable_response = self.dashboard.EnableRobot()
        if self._response_code(enable_response) != 0:
            raise RuntimeError(f"EnableRobot failed: {enable_response!r}")
        speed_response = self.dashboard.SpeedFactor(config.STARTUP_SPEED_FACTOR)
        if self._response_code(speed_response) != 0:
            raise RuntimeError(f"SpeedFactor failed: {speed_response!r}")

        now = time.monotonic()
        self.startup_hold_until = now + config.STARTUP_HOLD_SECONDS
        self.startup_slow_until = self.startup_hold_until + config.STARTUP_SLOW_SECONDS
        self.normal_speed_restored = False
        self._start_servo_loop()
        print("[safety] continuous ServoJ tracking authorized")
        return True

    def _start_servo_loop(self) -> None:
        if self._servo_thread is not None and self._servo_thread.is_alive():
            return
        if self.command_angles is None:
            raise RuntimeError("ServoJ cannot start before the initial pose is captured")
        self._stop_event.clear()
        self._servo_thread = threading.Thread(
            target=self._servo_loop,
            name="dobot-servoj-loop",
            daemon=True,
        )
        self._servo_thread.start()

    def update(self, result: TrackingResult) -> None:
        """Publish the newest vision result without sending a robot command."""
        now = time.monotonic()
        valid = result.locked and result.tracking and result.center is not None

        with self._state_lock:
            if valid:
                self.lost_frame_count = 0
                self.err_x_buf.append(float(result.error_x))
                self.err_y_buf.append(float(result.error_y))
                self.latest_err_x = sum(self.err_x_buf) / len(self.err_x_buf)
                self.latest_err_y = sum(self.err_y_buf) / len(self.err_y_buf)
                self.latest_similarity = float(result.similarity)
                self.latest_source = getattr(result, "tracking_source", "recognition")
                self.latest_timestamp = float(result.timestamp)
                self.target_valid = True
            else:
                self.lost_frame_count += 1
                self.target_valid = False
                self.latest_err_x = 0.0
                self.latest_err_y = 0.0
                if self.lost_frame_count > config.MAX_LOST_FRAMES:
                    self.err_x_buf.clear()
                    self.err_y_buf.clear()

        if now - self.last_vision_log >= 1.0 / config.VISION_LOG_HZ:
            state = "LOCKED" if valid else "LOST"
            print(
                f"[vision input] {state} error=({result.error_x:+d},{result.error_y:+d}) "
                f"similarity={result.similarity:.3f}"
            )
            self.last_vision_log = now

    def _snapshot_vision(self):
        with self._state_lock:
            return (
                self.target_valid,
                self.latest_err_x,
                self.latest_err_y,
                self.latest_similarity,
                self.latest_source,
                self.latest_timestamp,
                self.lost_frame_count,
            )

    def _desired_velocities(self, now_wall: float, now_mono: float):
        valid, err_x, err_y, similarity, source, timestamp, lost_count = (
            self._snapshot_vision()
        )
        fresh = valid and now_wall - timestamp <= config.DATA_TIMEOUT

        if self.startup_hold_until is not None and now_mono < self.startup_hold_until:
            return 0.0, 0.0, err_x, err_y, similarity, source, "startup-hold"
        if not fresh:
            return 0.0, 0.0, err_x, err_y, similarity, source, f"hold-lost-{lost_count}"

        control_x = 0.0 if abs(err_x) <= config.DEAD_ZONE_X else err_x
        control_y = 0.0 if abs(err_y) <= config.DEAD_ZONE_Y else err_y

        # Preserve the physically verified current build directions. The old
        # controller used J1_OUTPUT_SIGN followed by one final inversion.
        yaw_output_sign = -float(config.J1_OUTPUT_SIGN)
        desired_yaw = config.SERVO_KP_YAW_VEL * control_x * yaw_output_sign
        desired_pitch = config.SERVO_KP_PITCH_VEL * control_y

        yaw_limit = config.SERVO_MAX_YAW_VEL
        pitch_limit = config.SERVO_MAX_PITCH_VEL
        phase = "tracking"
        if self.startup_slow_until is not None and now_mono < self.startup_slow_until:
            yaw_limit = min(yaw_limit, config.SERVO_STARTUP_MAX_YAW_VEL)
            pitch_limit = min(pitch_limit, config.SERVO_STARTUP_MAX_PITCH_VEL)
            phase = "startup-slow"
        if source == "tracker":
            yaw_limit *= config.TRACKER_SPEED_SCALE
            pitch_limit *= config.TRACKER_SPEED_SCALE

        return (
            self._clamp(desired_yaw, -yaw_limit, yaw_limit),
            self._clamp(desired_pitch, -pitch_limit, pitch_limit),
            err_x,
            err_y,
            similarity,
            source,
            phase,
        )

    def _servo_loop(self) -> None:
        period = 1.0 / config.SERVO_CONTROL_HZ
        previous_t = time.monotonic()
        next_t = previous_t

        try:
            while not self._stop_event.is_set():
                now_mono = time.monotonic()
                wait_s = next_t - now_mono
                if wait_s > 0.0:
                    self._stop_event.wait(wait_s)
                    if self._stop_event.is_set():
                        break
                    now_mono = time.monotonic()

                dt = self._clamp(now_mono - previous_t, 0.001, 0.10)
                previous_t = now_mono
                next_t = max(next_t + period, now_mono)

                (
                    desired_yaw,
                    desired_pitch,
                    err_x,
                    err_y,
                    similarity,
                    source,
                    phase,
                ) = self._desired_velocities(time.time(), now_mono)

                if phase.startswith("hold") or phase == "startup-hold":
                    self.yaw_velocity = 0.0
                    self.pitch_velocity = 0.0
                else:
                    self.yaw_velocity = self._approach(
                        self.yaw_velocity,
                        desired_yaw,
                        config.SERVO_MAX_YAW_ACCEL * dt,
                    )
                    self.pitch_velocity = self._approach(
                        self.pitch_velocity,
                        desired_pitch,
                        config.SERVO_MAX_PITCH_ACCEL * dt,
                    )

                target = list(self.command_angles)
                target[0] += self.yaw_velocity * dt
                target[1] += self.pitch_velocity * dt

                initial = self.startup_initial_angles
                j1_min = initial[0] - config.J1_TRACK_WINDOW_DEG
                j1_max = initial[0] + config.J1_TRACK_WINDOW_DEG
                j2_min = initial[1] - config.J2_TRACK_WINDOW_DEG
                j2_max = initial[1] + config.J2_TRACK_WINDOW_DEG
                clamped_j1 = self._clamp(target[0], j1_min, j1_max)
                clamped_j2 = self._clamp(target[1], j2_min, j2_max)
                if clamped_j1 != target[0]:
                    self.yaw_velocity = 0.0
                if clamped_j2 != target[1]:
                    self.pitch_velocity = 0.0
                target[0] = clamped_j1
                target[1] = clamped_j2

                # J3-J6 are copied from the captured pose and never receive a
                # tracking increment. This includes the unavailable J6 roll.
                for index in range(2, 6):
                    target[index] = initial[index]

                if self.simulation:
                    self.sim_angles = list(target)
                    self.actual_angles = list(target)
                else:
                    if (
                        not self.normal_speed_restored
                        and self.startup_slow_until is not None
                        and now_mono >= self.startup_slow_until
                    ):
                        self.dashboard.SpeedFactor(config.REAL_SPEED_FACTOR)
                        self.normal_speed_restored = True
                    response = self.dashboard.ServoJ(
                        *target,
                        t=config.SERVO_COMMAND_T,
                        aheadtime=config.SERVO_AHEADTIME,
                        gain=config.SERVO_GAIN,
                    )
                    if self._response_code(response) != 0:
                        raise RuntimeError(f"ServoJ rejected: {response!r}")

                    if now_mono - self.last_actual_read >= 1.0:
                        self.actual_angles = self._read_current_angles()
                        self.last_actual_read = now_mono

                self.command_angles = list(target)

                if now_mono - self.last_servo_log >= 1.0 / config.SERVO_LOG_HZ:
                    actual_j1 = self.actual_angles[0] if self.actual_angles else float("nan")
                    actual_j2 = self.actual_angles[1] if self.actual_angles else float("nan")
                    print(
                        f"[servo] phase={phase} source={source} "
                        f"err=({err_x:+.0f},{err_y:+.0f}) sim={similarity:.3f} "
                        f"vel=({self.yaw_velocity:+.2f},{self.pitch_velocity:+.2f})deg/s "
                        f"target=({target[0]:.2f},{target[1]:.2f}) "
                        f"actual=({actual_j1:.2f},{actual_j2:.2f})"
                    )
                    self.last_servo_log = now_mono
        except Exception as exc:
            self._worker_error = str(exc)
            print(f"[error] ServoJ loop stopped: {exc}")
            if not self.simulation and self.dashboard is not None:
                try:
                    self.dashboard.Stop()
                except Exception:
                    pass
            self._stop_event.set()

    def disconnect(self) -> None:
        self._stop_event.set()
        if self._servo_thread is not None:
            self._servo_thread.join(timeout=2.0)

        if self.dashboard is not None and not self.simulation:
            try:
                self.dashboard.Stop()
            except Exception:
                pass
            try:
                self.dashboard.close()
                self.dashboard.socket_dobot = 0
            except Exception:
                pass
            print("[robot] ServoJ stopped; robot remains enabled")
