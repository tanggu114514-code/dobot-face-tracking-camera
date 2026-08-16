"""Camera-only test for target recognition and eye-line deskewing."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision import Vision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["webcam", "realsense"], default="realsense")
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()

    vision = Vision(source=args.source, camera_index=args.camera_index)
    print("[safety] Vision-only mode. No robot connection or command is used.")
    print("Press ESC or Q to quit.")
    try:
        empty_frames = 0
        while True:
            frame = vision.read()
            if frame is None:
                empty_frames += 1
                if empty_frames == 1:
                    print("[info] waiting for camera frames...")
                if empty_frames > 200:
                    print("[error] camera frame unavailable after 10 seconds")
                    return 2
                time.sleep(0.05)
                continue

            empty_frames = 0
            result = vision.update(frame)
            status = "target LOCKED" if result.locked else "target LOST"
            color = (0, 0, 255) if result.locked else (180, 180, 180)
            cv2.putText(frame, status, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(
                frame,
                (
                    f"error=({result.error_x:+d},{result.error_y:+d}) "
                    f"deskew={vision.deskewer.angle_deg:+.2f} deg"
                ),
                (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
            )
            if result.bbox:
                x1, y1, x2, y2 = result.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            cv2.imshow("Target Vision Deskew Test - No Robot", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                return 0
    finally:
        vision.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
