# Architecture

## Data flow

```text
Camera frame
  -> YuNet face detection
  -> SFace comparison against local reference images
  -> target center and optional RealSense camera-space depth
  -> smoothing and target-state logic
  -> TrackingResult
  -> dead zone and proportional velocity control
  -> acceleration and joint-window limits
  -> 20 Hz ServoJ J1/J2 commands
```

## Modules

- `src/face_engine.py`: camera sources, model download, YuNet and SFace.
- `src/vision.py`: target state, partial-face fallback, smoothing and 3-D output.
- `src/robot_controller.py`: simulation, arming, fixed-rate ServoJ loop and limits.
- `src/main_integrate.py`: user interface, enrollment and module integration.
- `src/face_db.py`: local-only enrollment storage and optional LBPH training.
- `src/config.py`: thresholds, rates, limits and robot network settings.

## Coordinate contract

```text
error_x = image_center_x - target_center_x
error_y = configured_center_y - target_center_y
```

J1 output direction is hardware-dependent. Run the calibration tool before
real tracking rather than guessing signs.

## Current limitations

- J1 and J2 are the only commanded tracking joints.
- J3 through J6 hold their startup angles.
- Depth is measured but does not command Cartesian or joint movement.
- No hand-eye calibration or camera-to-base transform is included.
- CSRT is a short continuity aid, not an identity recognizer.
