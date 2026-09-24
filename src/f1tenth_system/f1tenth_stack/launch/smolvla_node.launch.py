from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
import os
import shutil


def generate_launch_description():
    stack = get_package_share_directory("f1tenth_stack")
    params = os.path.join(stack, "config", "smolvla.yaml")
    root = Path(__file__).resolve().parent.parent
    py = shutil.which("python3")

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(stack, "launch", "camera_test.launch.py")
        )
    )

    policy = ExecuteProcess(
        cmd=[
            py,
            str(root / "f1tenth_stack" / "smolvla_openloop.py"),
            "--ros-args",
            "--params-file",
            params,
        ],
        output="screen",
        emulate_tty=True,
    )
    adapter = ExecuteProcess(
        cmd=[
            py,
            str(root / "f1tenth_stack" / "smolvla_to_drive.py"),
            "--ros-args",
            "--params-file",
            params,
        ],
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription([policy, adapter])
    # return LaunchDescription([camera, policy, adapter])
