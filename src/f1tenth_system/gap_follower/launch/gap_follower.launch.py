from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    params_file = os.path.join(
        get_package_share_directory('gap_follower'),
        'config',
        'gap_follower_params.yaml'
    )

    gap_follower_node = Node(
        package='gap_follower',
        executable='gap_follower_node',
        name='gap_follower_node',
        output='screen',
        parameters=[params_file]
    )

    path_follower_node = Node(
        package='gap_follower',
        executable='path_follower_node',
        name='path_follower_node',
        output='screen',
        parameters=[params_file]
    )

    return LaunchDescription([
        # gap_follower_node,
        path_follower_node,
    ])
