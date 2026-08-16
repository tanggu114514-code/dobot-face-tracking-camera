"""Vision plus continuous ServoJ tracking, simulated unless explicitly armed."""

import argparse
import os
import shutil
import time

import cv2

import config
import face_db
from robot_controller import RobotController, TrackingResult
from vision import Vision


SAMPLE_TOTAL = 20
SAMPLE_INTERVAL_SECONDS = 0.25


def make_hold_result(frame) -> TrackingResult:
    """Tell the ServoJ loop to decelerate to zero and hold its current pose."""
    height, width = frame.shape[:2]
    return TrackingResult(
        locked=False,
        tracking=False,
        target_name="",
        similarity=0.0,
        bbox=None,
        center=None,
        error_x=0,
        error_y=0,
        frame_width=width,
        frame_height=height,
        timestamp=time.time(),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["webcam", "realsense"], default="webcam")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--real-robot",
        action="store_true",
        help="Explicitly request real continuous J1/J2 ServoJ control.",
    )
    args = parser.parse_args()

    if args.real_robot:
        print("===== REAL SERVOJ ROBOT REQUESTED =====")
        print("This test continuously controls J1/J2; J3-J6 are held unchanged.")
        print("Clear the workspace and keep the emergency stop reachable.")
        first = input("Type REAL to continue, or anything else to cancel: ").strip().upper()
        if first != "REAL":
            print("[safety] cancelled; no robot command was sent")
            return 0
        config.SIMULATION_MODE = False
    else:
        config.SIMULATION_MODE = True

    # On-site LBPH learning channel. Normal tracking below remains unchanged.
    creator = getattr(getattr(cv2, "face", None), "LBPHFaceRecognizer_create", None)
    if not callable(creator):
        print("[error] LBPH is unavailable; install opencv-contrib-python")
        return 2
    face_recognizer = creator()
    if os.path.exists(face_db.MODEL_SAVE):
        face_recognizer.read(face_db.MODEL_SAVE)
        print(f"[learning] loaded LBPH model: {face_db.MODEL_SAVE}")
    else:
        print("[learning] no LBPH model yet; press S to enroll the first target")

    vision = Vision(source=args.source, camera_index=args.camera_index)
    robot = RobotController()
    try:
        robot.connect()
    except Exception as exc:
        print(f"[error] robot connection failed: {exc}")
        vision.release()
        return 2

    if args.real_robot and robot.simulation:
        print("[safety] 真机模式未建立，程序退出；没有发送任何机器人命令。")
        vision.release()
        return 2

    if args.real_robot and not robot.simulation:
        if not robot.enable_real_robot():
            robot.disconnect()
            return 0
    if robot.simulation:
        print("===== VISION + SERVOJ SIMULATION =====")
        print("No real robot command will be sent. Press ESC to quit.")
    else:
        print("===== CONTINUOUS REAL J1/J2 SERVO TRACKING =====")
        print("Press ESC to stop; keep the emergency stop reachable.")

    learning_mode = False
    new_target_id = None
    sample_count = 0
    current_target_name = ""
    target_dir = None
    last_sample_time = 0.0

    try:
        empty_frames = 0
        while True:
            frame = vision.read()
            if frame is None:
                empty_frames += 1
                if empty_frames == 1:
                    print("[info] waiting for the first RealSense frame...")
                if empty_frames > 200:
                    print("[error] camera frame unavailable after 10 seconds")
                    break
                time.sleep(0.05)
                continue

            empty_frames = 0

            # Learning branch: collect one face and do not publish movement targets.
            if learning_mode:
                robot.update(make_hold_result(frame))
                frame_copy = frame.copy()
                enrollment_faces = vision.engine.detect_enrollment_faces(frame_copy)

                if len(enrollment_faces) == 1:
                    face = enrollment_faces[0]
                    x1, y1, x2, y2 = (int(value) for value in face.box)
                    cv2.rectangle(
                        frame_copy,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )
                    now = time.monotonic()
                    if (
                        sample_count < SAMPLE_TOTAL
                        and now - last_sample_time >= SAMPLE_INTERVAL_SECONDS
                    ):
                        width = x2 - x1
                        height = y2 - y1
                        padding = int(round(max(width, height) * 0.30))
                        crop_x1 = max(0, x1 - padding)
                        crop_y1 = max(0, y1 - padding)
                        crop_x2 = min(frame_copy.shape[1], x2 + padding)
                        crop_y2 = min(frame_copy.shape[0], y2 + padding)
                        face_crop = frame_copy[crop_y1:crop_y2, crop_x1:crop_x2]
                        sample_path = os.path.join(
                            target_dir,
                            f"sample_{sample_count:02d}.jpg",
                        )
                        if cv2.imwrite(sample_path, face_crop):
                            sample_count += 1
                            last_sample_time = now
                elif len(enrollment_faces) == 0:
                    cv2.putText(
                        frame_copy,
                        "Show exactly one face",
                        (30, 78),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),
                        2,
                    )
                else:
                    cv2.putText(
                        frame_copy,
                        "Multiple faces: sampling paused",
                        (30, 78),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),
                        2,
                    )

                cv2.putText(
                    frame_copy,
                    f"ENROLL {current_target_name}: {sample_count}/{SAMPLE_TOTAL}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame_copy,
                    "Turn slowly; ESC cancels enrollment",
                    (30, frame_copy.shape[0] - 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Learning Channel - New Target", frame_copy)

                if sample_count >= SAMPLE_TOTAL:
                    print(f"\n[learning] {current_target_name} captured; loading SFace...")
                    try:
                        feature_count = vision.set_target(
                            current_target_name,
                            target_dir,
                        )
                    except Exception as exc:
                        print(f"[learning] SFace target activation failed: {exc}")
                        print("[learning] old tracking target remains active")
                    else:
                        try:
                            face_db.save_label_map(new_target_id, current_target_name)
                        except Exception as exc:
                            print(f"[learning] database label save failed: {exc}")
                        else:
                            try:
                                face_recognizer = face_db.train_update_model()
                            except Exception as exc:
                                print(f"[learning] optional LBPH training failed: {exc}")
                        print(
                            f"===== {current_target_name} is now the active SFace target "
                            f"({feature_count} references) =====\n"
                        )
                    learning_mode = False
                    cv2.waitKey(1)
                    cv2.destroyWindow("Learning Channel - New Target")
                    continue

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    learning_mode = False
                    shutil.rmtree(target_dir, ignore_errors=True)
                    print("[learning] cancelled; temporary samples removed\n")
                    cv2.destroyWindow("Learning Channel - New Target")
                continue

            # Normal tracking branch: original vision and robot logic.
            result = vision.update(frame)
            robot.update(result)

            tracking_source = vision.get_tracking_source()
            target_name = result.target_name or vision.target_name
            if tracking_source == "confirming":
                text = f"{target_name} CONFIRMING"
            elif result.locked and tracking_source == "tracker":
                text = f"{target_name} TRACKED"
            else:
                text = (
                    f"{target_name} LOCKED" if result.locked else f"{target_name} LOST"
                )
            color = (0, 0, 255) if result.locked else (180, 180, 180)
            cv2.putText(frame, text, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(
                frame,
                f"error_x={result.error_x:+d} error_y={result.error_y:+d}",
                (24, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            camera_data = vision.get_last_3d()
            depth_m = camera_data.get("depth_m")
            depth_text = "n/a" if depth_m is None else f"{float(depth_m):.3f}m"
            cv2.putText(
                frame,
                f"similarity={result.similarity:.3f} depth={depth_text}",
                (24, 108),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            for detection in vision.get_last_detections():
                x1, y1, x2, y2 = detection["bbox"]
                is_target = bool(detection["is_target"])
                box_color = (0, 0, 255) if is_target else (255, 0, 0)
                source = detection.get("source", "recognition")
                if is_target and source == "tracker":
                    label = f"{target_name} TRACK"
                elif is_target and source == "hold":
                    label = f"{target_name} HOLD"
                elif is_target and source == "confirming":
                    label = f"{target_name} CONFIRM"
                else:
                    label = target_name if is_target else "FACE"
                label += f" {float(detection['similarity']):.3f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2,
                )

            cv2.imshow("Continuous ServoJ Face Tracking", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                robot.update(make_hold_result(frame))
                print("\n===== enter on-site target learning channel =====")
                current_target_name = input("New target name: ").strip()
                if not current_target_name:
                    print("[learning] empty name; cancelled")
                    continue
                new_target_id, _ = face_db.get_new_id()
                target_dir = os.path.join(face_db.DB_PATH, f"id_{new_target_id}")
                try:
                    os.makedirs(target_dir, exist_ok=False)
                except FileExistsError:
                    print(f"[learning] target folder already exists: {target_dir}")
                    continue
                sample_count = 0
                last_sample_time = 0.0
                learning_mode = True
                cv2.destroyWindow("Continuous ServoJ Face Tracking")
                print(
                    f"[learning] capturing {SAMPLE_TOTAL} samples for "
                    f"{current_target_name}; turn your head slowly"
                )
            elif key == 27:
                break
    finally:
        vision.release()
        robot.disconnect()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
