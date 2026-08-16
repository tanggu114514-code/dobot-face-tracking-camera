"""Read-only DOBOT status check. This script sends no motion command."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vendor.dobot_api import DobotApiDashboard
import config


MODE_NAMES = {
    1: "INIT",
    2: "BRAKE_OPEN",
    3: "POWEROFF",
    4: "DISABLED",
    5: "ENABLED_IDLE",
    6: "BACKDRIVE",
    7: "RUNNING",
    8: "SINGLE_MOVE",
    9: "ERROR",
    10: "PAUSE",
    11: "COLLISION",
}


def main() -> int:
    print("[safety] Read-only check: no EnableRobot, Stop, ClearError, or motion command.")
    dashboard = DobotApiDashboard(config.ROBOT_IP, config.DASHBOARD_PORT)
    try:
        mode = dashboard.RobotMode()
        errors = dashboard.GetErrorID()
        angles = dashboard.GetAngle()
        current_command_id = dashboard.GetCurrentCommandID()
        print(f"RobotMode raw: {mode}")
        for number, name in MODE_NAMES.items():
            if f"{{{number}}}" in str(mode):
                print(f"RobotMode decoded: {number} ({name})")
                break
        print(f"GetErrorID: {errors}")
        print(f"GetAngle: {angles}")
        print(f"GetCurrentCommandID: {current_command_id}")
    finally:
        dashboard.close()
        # The bundled SDK destructor calls close() again. Mark this socket as
        # already closed to avoid a harmless WinError 10038 on interpreter exit.
        dashboard.socket_dobot = 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
