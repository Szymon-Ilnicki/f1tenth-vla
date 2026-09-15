from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    pkg = FindPackageShare('f1tenth_stack')

    slam_params = DeclareLaunchArgument(
        'slam_params_file',
        default_value=PathJoinSubstitution([pkg, 'config', 'tiny_slam.yaml']),
        description='Full path to slam config'
    )

    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false'
    )

    slam_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='both',
        parameters=[
            LaunchConfiguration('slam_params_file'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        remappings=[('/scan', '/scan')]
    )

    return LaunchDescription([slam_params, use_sim_time, slam_node])
