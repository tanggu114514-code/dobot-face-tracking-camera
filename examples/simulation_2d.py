"""Camera-free 2-D visual tracking simulation."""

import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_controller import RobotController, TrackingResult


WIDTH, HEIGHT = 848, 480
FACE_W, FACE_H = 220, 260


def make_result(frame_index: int, robot_j1: float) -> TrackingResult:
    # The target moves in the virtual world. The simulated J1 rotation changes
    # its image position, creating a closed loop instead of an open-loop test.
    phase = frame_index / 45.0
    world_offset_x = 210 * math.sin(phase)
    camera_correction_x = 12.0 * robot_j1
    center_x = int(WIDTH / 2 + world_offset_x + camera_correction_x)
    center_y = int(HEIGHT / 2 + 35 * math.sin(phase * 0.7))
    bbox = (
        center_x - FACE_W // 2,
        center_y - FACE_H // 2,
        center_x + FACE_W // 2,
        center_y + FACE_H // 2,
    )

    # Required convention: frame center x - target center x.
    return TrackingResult(
        locked=True,
        tracking=True,
        target_name="target",
        similarity=0.70,
        bbox=bbox,
        center=(center_x, center_y),
        error_x=WIDTH // 2 - center_x,
        error_y=HEIGHT // 2 - center_y,
        frame_width=WIDTH,
        frame_height=HEIGHT,
        timestamp=time.time(),
    )


def main() -> int:
    robot = RobotController()
    robot.connect()
    print("===== 2-D camera-free simulation =====")
    print("A virtual target moves left and right; no camera or robot is used.")
    print("Press ESC to quit.")

    try:
        for frame_index in range(900):
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            frame[:] = (35, 35, 35)
            current_j1 = 0.0 if robot.last_target_j1 is None else robot.last_target_j1
            result = make_result(frame_index, current_j1)
            robot.update(result)

            cv2.drawMarker(
                frame,
                (WIDTH // 2, HEIGHT // 2),
                (255, 255, 255),
                cv2.MARKER_CROSS,
                24,
                2,
            )
            x1, y1, x2, y2 = result.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "target (virtual)", (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"error_x={result.error_x:+d}", (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "SIMULATION ONLY", (24, HEIGHT - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)

            cv2.imshow("2-D Tracking Simulation", frame)
            if cv2.waitKey(33) & 0xFF == 27:
                break
    finally:
        robot.disconnect()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
