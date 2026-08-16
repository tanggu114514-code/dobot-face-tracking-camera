"""Safely measure the physical J1-to-image horizontal direction.

This tool does not run the face-tracking controller. It moves J1 once by a
small angle, measures the target face center before and after the move, then
prints the only J1_OUTPUT_SIGN value that should be used by the controller.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import statistics
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config
import face_engine as face_demo
from robot_controller import RobotController
from vision import Vision


RESULT_PATH = ROOT / "data" / "yaw_calibration_result.json"

CALIBRATION_SPEED_FACTOR = 5
DEFAULT_STEP_DEG = 0.30
MIN_STEP_DEG = 0.10
MAX_STEP_DEG = 0.50
SAMPLE_COUNT = 20
SAMPLE_TIMEOUT_SECONDS = 12.0
MOVE_TIMEOUT_SECONDS = 6.0
MIN_ACTUAL_J1_MOVE_DEG = 0.08
MIN_PIXEL_SHIFT = 2.0
MAX_STABLE_SAMPLE_SPREAD_PX = 8.0


def infer_output_sign(pixel_shift: float, joint_delta: float) -> int:
    """Return the sign that makes the image-space J1 loop negative feedback."""
    if abs(joint_delta) < MIN_ACTUAL_J1_MOVE_DEG:
        raise ValueError("J1 actual movement is too small to calibrate")
    if abs(pixel_shift) < MIN_PIXEL_SHIFT:
        raise ValueError("face pixel movement is too small to calibrate")
    return 1 if pixel_shift / joint_delta > 0.0 else -1


def wait_for_j1(robot: RobotController, target_j1: float) -> list[float]:
    """Wait until measured J1 reaches the tiny calibration target."""
    deadline = time.time() + MOVE_TIMEOUT_SECONDS
    stable_reads = 0
    last_angles = robot._read_current_angles()
    while time.time() < deadline:
        last_angles = robot._read_current_angles()
        if abs(last_angles[0] - target_j1) <= 0.04:
            stable_reads += 1
            if stable_reads >= 3:
                return last_angles
        else:
            stable_reads = 0
        time.sleep(0.08)
    raise RuntimeError(
        f"J1 did not reach {target_j1:.3f} deg; last reading was "
        f"{last_angles[0]:.3f} deg"
    )


def capture_face_x(vision: Vision, phase: str) -> tuple[float, float]:
    """Wait for a stable window, then return median x and similarity."""
    samples: deque[float] = deque(maxlen=SAMPLE_COUNT)
    similarities: deque[float] = deque(maxlen=SAMPLE_COUNT)
    deadline = time.time() + SAMPLE_TIMEOUT_SECONDS
    last_spread = float("inf")

    while time.time() < deadline:
        frame = vision.read()
        if frame is None:
            time.sleep(0.03)
            continue

        faces = vision.engine.recognize(frame, config.LOCK_THRESHOLD)
        target = next((item for item in faces if item.is_target), None)
        display = frame.copy()
        h, w = display.shape[:2]
        cv2.line(display, (w // 2, 0), (w // 2, h - 1), (0, 255, 255), 1)

        if target is not None:
            x1, y1, x2, y2 = target.box
            cx, cy = target.center
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(display, (cx, cy), 4, (0, 255, 255), -1)
            samples.append(float(cx))
            similarities.append(float(target.similarity))
            last_spread = max(samples) - min(samples) if len(samples) > 1 else 0.0
            status = (
                f"{phase}: keep still  stable={len(samples)}/{SAMPLE_COUNT} "
                f"spread={last_spread:.1f}px x={cx} sim={target.similarity:.3f}"
            )
            if (
                len(samples) == SAMPLE_COUNT
                and last_spread <= MAX_STABLE_SAMPLE_SPREAD_PX
            ):
                cv2.imshow("J1 Direction Calibration", display)
                cv2.waitKey(1)
                return statistics.median(samples), statistics.median(similarities)
        else:
            samples.clear()
            similarities.clear()
            status = f"{phase}: target not locked - face camera and keep still"

        cv2.putText(
            display,
            status,
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
        )
        cv2.imshow("J1 Direction Calibration", display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            raise KeyboardInterrupt("operator cancelled with ESC")
        time.sleep(0.02)

    raise RuntimeError(
        f"No stable face window during {phase}; latest spread was "
        f"{last_spread:.1f} px. Wait for the robot and camera to settle, then rerun."
    )


def choose_step(initial_j1: float, requested_step: float) -> float:
    """Use one tiny positive multi-turn J1 step.

    The robot reported startup J1 values outside +/-60 degrees, so the old
    absolute software bounds cannot be used to choose this local test step.
    """
    del initial_j1
    return requested_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure the correct physical J1_OUTPUT_SIGN without auto tracking."
    )
    parser.add_argument("--source", choices=["webcam", "realsense"], default="realsense")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--step-deg", type=float, default=DEFAULT_STEP_DEG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    step_size = abs(float(args.step_deg))
    if not MIN_STEP_DEG <= step_size <= MAX_STEP_DEG:
        print(
            f"[error] --step-deg must be between {MIN_STEP_DEG:.2f} "
            f"and {MAX_STEP_DEG:.2f} degrees"
        )
        return 2

    print("===== J1 DIRECTION CALIBRATION =====")
    print("This is NOT automatic face tracking.")
    print(f"J1 will move once by at most {step_size:.2f} deg at 5% speed, then return.")
    print("J2-J6 will be copied from their measured angles and will not be changed.")
    print("Clear the workspace and keep the emergency stop reachable.")
    if input("Type CALIBRATE to connect: ").strip().upper() != "CALIBRATE":
        print("[safety] cancelled before robot connection")
        return 0

    # Direction calibration must use stable, unrotated pixels. Dynamic Hough
    # leveling is intentionally disabled only inside this process.
    face_demo.AUTO_FIX_HORIZONTAL = False
    config.SIMULATION_MODE = False

    vision: Vision | None = None
    robot: RobotController | None = None
    initial_angles: list[float] | None = None
    moved = False

    try:
        vision = Vision(source=args.source, camera_index=args.camera_index)
        robot = RobotController()
        robot.connect()
        if robot.simulation or robot.dashboard is None:
            raise RuntimeError("real robot connection was not established")

        if not robot.enable_real_robot():
            return 0

        print("\nStand still with the target clearly visible near the middle of the image.")
        baseline_x, baseline_similarity = capture_face_x(vision, "BEFORE")
        initial_angles = robot._read_current_angles()
        test_step = choose_step(initial_angles[0], step_size)
        target_angles = list(initial_angles)
        target_angles[0] = initial_angles[0] + test_step

        print(
            f"\n[ready] target x={baseline_x:.1f}, similarity={baseline_similarity:.3f}, "
            f"J1={initial_angles[0]:.3f} deg"
        )
        print(
            f"[ready] Next move: J1 {test_step:+.3f} deg at "
            f"{CALIBRATION_SPEED_FACTOR}% speed. Keep your head still."
        )
        if input("Type MOVE to perform the tiny test: ").strip().upper() != "MOVE":
            print("[safety] cancelled before movement")
            return 0

        robot.dashboard.SpeedFactor(CALIBRATION_SPEED_FACTOR)
        response = robot.dashboard.MovJ(*target_angles, 1)
        print(f"[robot] calibration MovJ response: {response}")
        moved = True
        reached_angles = wait_for_j1(robot, target_angles[0])
        time.sleep(0.35)

        after_x, after_similarity = capture_face_x(vision, "AFTER")
        measured_angles = robot._read_current_angles()
        actual_joint_delta = measured_angles[0] - initial_angles[0]
        pixel_shift = after_x - baseline_x
        recommended_sign = infer_output_sign(pixel_shift, actual_joint_delta)

        result = {
            "timestamp": time.time(),
            "source": args.source,
            "baseline_face_x": baseline_x,
            "after_face_x": after_x,
            "pixel_shift": pixel_shift,
            "baseline_similarity": baseline_similarity,
            "after_similarity": after_similarity,
            "initial_j1_deg": initial_angles[0],
            "reached_j1_deg": reached_angles[0],
            "measured_j1_deg": measured_angles[0],
            "actual_j1_delta_deg": actual_joint_delta,
            "pixels_per_positive_j1_degree": pixel_shift / actual_joint_delta,
            "recommended_J1_OUTPUT_SIGN": recommended_sign,
            "current_J1_OUTPUT_SIGN": config.J1_OUTPUT_SIGN,
        }
        RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

        print("\n===== CALIBRATION RESULT =====")
        print(f"Face x: {baseline_x:.1f} -> {after_x:.1f}  shift={pixel_shift:+.1f} px")
        print(
            f"J1: {initial_angles[0]:.3f} -> {measured_angles[0]:.3f}  "
            f"actual delta={actual_joint_delta:+.3f} deg"
        )
        print(f"Recommended: J1_OUTPUT_SIGN = {recommended_sign:+d}")
        if recommended_sign == config.J1_OUTPUT_SIGN:
            print("Current config already matches this physical calibration.")
        else:
            print(
                f"Current config is {config.J1_OUTPUT_SIGN:+d}; change only "
                f"J1_OUTPUT_SIGN to {recommended_sign:+d} before tracking."
            )
        print(f"Result saved to: {RESULT_PATH}")
        return 0

    except KeyboardInterrupt:
        print("\n[safety] calibration cancelled by operator")
        return 130
    except Exception as exc:
        print(f"\n[error] calibration failed: {exc}")
        print("No direction setting was changed.")
        return 2
    finally:
        if robot is not None and initial_angles is not None and moved:
            try:
                print("[robot] returning J1 to its pre-test angle...")
                current = robot._read_current_angles()
                return_angles = list(current)
                return_angles[0] = initial_angles[0]
                robot.dashboard.SpeedFactor(CALIBRATION_SPEED_FACTOR)
                robot.dashboard.MovJ(*return_angles, 1)
                wait_for_j1(robot, initial_angles[0])
                print("[robot] J1 returned to the pre-test angle")
            except Exception as exc:
                print(f"[warning] automatic J1 return failed: {exc}")
                print("Use the emergency stop or DobotStudio manual controls if needed.")
        if robot is not None:
            robot.disconnect()
        if vision is not None:
            vision.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
