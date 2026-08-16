# simulation_demo.py 仿真调试脚本，用来单独验证控制模块全部逻辑
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robot_controller import RobotController, TrackingResult
import config

if __name__ == "__main__":
    if not config.SIMULATION_MODE:
        print("[safety] 当前配置已关闭仿真，simulation_demo.py 不执行任何机器人测试。")
        print("[safety] 请使用 main_integrate.py --real-robot，并完成现场安全确认。")
        raise SystemExit(2)

    # 初始化控制器，默认仿真模式
    robot = RobotController()
    robot.connect()

    # 模拟4组测试场景，循环测试全部安全逻辑
    test_scenes = [
        # 场景1：人脸在画面右侧 error_x=+140 正常跟踪
        TrackingResult(
            locked=True, tracking=True, target_name="target", similarity=0.7,
            bbox=(400,200,700,400), center=(563,277),
            error_x=140, error_y=36, frame_width=848, frame_height=480,
            timestamp=time.time()
        ),
        # 场景2：人脸在死区内，应当HOLD
        TrackingResult(
            locked=True, tracking=True, target_name="target", similarity=0.7,
            bbox=(400,200,700,400), center=(420,240),
            error_x=15, error_y=36, frame_width=848, frame_height=480,
            timestamp=time.time()
        ),
        # 场景3：目标丢失 locked=False，应当HOLD
        TrackingResult(
            locked=False, tracking=False, target_name="target", similarity=0.3,
            bbox=None, center=None,
            error_x=140, error_y=36, frame_width=848, frame_height=480,
            timestamp=time.time()
        ),
        # 场景4：人脸在画面左侧 error_x=-120
        TrackingResult(
            locked=True, tracking=True, target_name="target", similarity=0.69,
            bbox=(100,200,350,400), center=(200,277),
            error_x=-120, error_y=36, frame_width=848, frame_height=480,
            timestamp=time.time()
        ),
    ]

    print("===== 仿真测试开始，仅运行一轮 =====")
for scene in test_scenes:
    scene.timestamp = time.time()
    robot.update(scene)
    time.sleep(1.2)
print("仿真测试全部执行完毕，程序结束")
