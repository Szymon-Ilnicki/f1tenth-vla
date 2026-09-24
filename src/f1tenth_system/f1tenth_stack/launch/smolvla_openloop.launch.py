from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    stack = get_package_share_directory("f1tenth_stack")
    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(stack, "launch", "camera_test.launch.py")
        )
    )

    params = os.path.join(stack, "config", "smolvla.yaml")
    script = os.path.expanduser(
        "~/szymon_ws/f1tenth-vla/src/f1tenth_system/f1tenth_stack/f1tenth_stack/smolvla_openloop.py"
    )
    py = os.path.expanduser("~/venv-smolvla/bin/python3")

    policy = ExecuteProcess(
        cmd=[py, script, "--ros-args", "--params-file", params],
        output="screen",
        emulate_tty=True,
    )
    return LaunchDescription([camera, policy])