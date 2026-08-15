# DOBOT Target-Face Camera Tracking

A privacy-clean research prototype that recognizes one enrolled target face and
keeps it near the image center with continuous DOBOT ServoJ control.

The public repository contains no enrolled faces, identity labels, local
calibration results, or trained biometric files. Simulation is the default.

## Current scope

- YuNet face detection and SFace identity matching.
- Multiple reference images for one active target.
- Short CSRT fallback when a face is partially outside the frame.
- Exponential center smoothing, dead zones, velocity limits and acceleration limits.
- Continuous 20 Hz ServoJ control for J1 yaw and J2 pitch.
- J3, J4, J5 and J6 remain at their captured startup angles.
- RealSense depth is reported in camera coordinates but does not command motion.
- Optional software-only eye-line deskew for the displayed image.

This is not calibrated 3-D person following. `point_base_m` remains unavailable
until hand-eye calibration and robot-pose transforms are implemented.

## Layout

```text
.
|-- src/            Main application and control modules
|-- src/vendor/     Vendored DOBOT TCP/IP compatibility module
|-- data/           Local enrollment data; ignored by Git
|-- models/         Downloaded YuNet/SFace models; ignored by Git
|-- examples/       Camera-free simulations
|-- tools/          Read-only status, direction calibration and vision tests
|-- docs/           Architecture, privacy and robot-safety notes
`-- tests/          Release hygiene tests
```

## Install

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

YuNet and SFace are downloaded from the official OpenCV Zoo repository the
first time the vision program starts.

## Test without a robot

Controller-only smoke test:

```powershell
python examples\simulation_demo.py
```

Computer webcam:

```powershell
python src\main_integrate.py --source webcam
```

RealSense camera with simulated robot:

```powershell
python src\main_integrate.py --source realsense
```

No robot command is sent unless `--real-robot` is explicitly supplied.

## Enroll a target locally

1. Start either camera-only command above.
2. Press `S` in the video window.
3. Enter a local display name in the terminal.
4. Keep exactly one consenting person visible and slowly turn their head.
5. The program captures 20 samples and activates the new SFace target.

Enrollment files are written under `data/` and are excluded by `.gitignore`.
Delete that directory's generated contents to erase local identities.

## Real robot

Read [docs/SAFETY.md](docs/SAFETY.md) before using this mode. Set the robot IP
for the current PowerShell session, review the conservative motion limits in
`src/config.py`, verify direction calibration, clear the workspace and keep the
emergency stop reachable. The address is intentionally not stored in source:

```powershell
$env:DOBOT_ROBOT_IP = "YOUR_ROBOT_IP"
```

```powershell
python src\main_integrate.py --source realsense --real-robot
```

The program requires `REAL` and then `E`. Connecting alone does not enable the
robot in this release build.

## Verification

```powershell
python -m compileall -q src examples tools tests
python -m unittest discover -s tests -v
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md),
[docs/PRIVACY.md](docs/PRIVACY.md), and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details.
