from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    stack = get_package_share_directory('f1tenth_stack')
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='models/track_v1',
        description='Path to the pretrained model folder'
    )
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
