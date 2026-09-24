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
    script = str(
        Path(__file__).resolve().parent.parent / "f1tenth_stack" / "smolvla_openloop.py"
    )
    py = shutil.which("python3")

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(stack, "launch", "camera_test.launch.py")
        )
    )

    policy = ExecuteProcess(
        cmd=[py, script, "--ros-args", "--params-file", params],
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription([policy])
    # return LaunchDescription([camera, policy])
