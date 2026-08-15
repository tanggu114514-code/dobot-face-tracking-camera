# Robot Safety

This software can command physical machinery. It is a research prototype, not
a certified safety controller.

## Before real movement

1. Use simulation mode first.
2. Confirm the correct robot model, firmware, TCP/IP mode and IP address.
3. Run `python tools/robot_status_check.py`; do not proceed from an alarm,
   collision, emergency-stop or unexpected mode.
4. Run `python tools/calibrate_yaw_direction.py --source realsense` and verify
   the physical J1 direction with a tiny movement.
5. Clear the entire workspace and remove loose cables and mounts.
6. Keep the emergency stop reachable and assign one operator to it.
7. Begin with conservative velocity, acceleration and joint-window limits.

## During operation

- Keep people outside the robot's reachable workspace.
- Stop immediately on unexpected direction, vibration, collision indication,
  camera loss, communication error or loose hardware.
- Do not disable collision detection to make this demo run.
- Tracker-only frames are less trustworthy than fresh recognition frames.

## Control boundaries

- `--real-robot` is required.
- The application asks for `REAL` before connecting and `E` before enabling.
- The robot address must be supplied through `DOBOT_ROBOT_IP`; it is not stored
  in the public source tree.
- J1/J2 remain inside windows relative to their measured startup angles.
- Loss or stale vision commands zero velocity and holds the current pose.
- Exiting sends `Stop()` but deliberately does not power off the arm.

An external emergency stop and the robot's built-in safety system remain the
primary safety mechanisms.
