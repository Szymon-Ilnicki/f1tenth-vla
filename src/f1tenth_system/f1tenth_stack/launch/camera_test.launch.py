from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    rs_launch = os.path.join(
        get_package_share_directory("realsense2_camera"),
        "launch",
        "rs_launch.py",
    )
    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(rs_launch),
                launch_arguments={
                    "enable_color": "true",
                    "enable_depth": "false",
                    "enable_infra1": "false",
                    "enable_infra2": "false",
                    "rgb_camera.color_profile": "640x480x15",
                }.items(),
            )
        ]
    )
