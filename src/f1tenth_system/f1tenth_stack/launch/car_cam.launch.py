from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

# RGB-only RealSense D435i for OpenVLA preview / later inference. Only useful for testing the camera
# enable_color=true
# enable_depth=false          # depth off to save USB/CPU on the Orin Nano
# rgb_camera.color_profile=640x480x15
# Topic: /camera/camera/color/image_raw

def generate_launch_description():
    car = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('f1tenth_stack'),
                'launch',
                'car_min.launch.py',
            )
        )
    )

    cam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch',
                'rs_launch.py',
            )
        ),
        launch_arguments={
            'enable_color': 'true',
            'enable_depth': 'false',
            'rgb_camera.color_profile': '640x480x15',
        }.items(),
    )

    return LaunchDescription([car, cam])