import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params_file = os.path.join(
        get_package_share_directory("global_planner"),
        "config",
        "planner_params.yaml",
    )

    node = Node(
        package="global_planner",
        executable="global_planner_map_node",
        name="global_planner_node",
        output="screen",
        parameters=[params_file],
    )

    return LaunchDescription([node])