from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    stack = get_package_share_directory('f1tenth_stack')

    car_cam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(stack, 'launch', 'car_cam.launch.py')
        )
    )

    smolvla = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(stack, 'launch', 'smolvla_node.launch.py')
        )
    )

    return LaunchDescription([
        car_cam,
        smolvla,
    ])
