from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg = get_package_share_directory('f1tenth_stack')
    joy_teleop_config = os.path.join(pkg, 'config', 'joy_teleop.yaml')
    vesc_config = os.path.join(pkg, 'config', 'vesc.yaml')
    mux_config = os.path.join(pkg, 'config', 'mux.yaml')

    ld = LaunchDescription([
        DeclareLaunchArgument('joy_config', default_value=joy_teleop_config),
        DeclareLaunchArgument('vesc_config', default_value=vesc_config),
        DeclareLaunchArgument('mux_config', default_value=mux_config),
    ])

    ld.add_action(Node(
        package='joy',
        executable='joy_node',
        name='joy',
        parameters=[LaunchConfiguration('joy_config')],
    ))
    ld.add_action(Node(
        package='joy_teleop',
        executable='joy_teleop',
        name='joy_teleop',
        parameters=[LaunchConfiguration('joy_config')],
    ))
    ld.add_action(Node(
        package='f1tenth_stack',
        executable='throttle_interpolator',
        name='throttle_interpolator',
        parameters=[LaunchConfiguration('vesc_config')],
    ))
    ld.add_action(Node(
        package='vesc_ackermann',
        executable='vesc_to_odom_node',
        name='vesc_to_odom_node',
        parameters=[LaunchConfiguration('vesc_config')],
    ))
    ld.add_action(Node(
        package='vesc_driver',
        executable='vesc_driver_node',
        name='vesc_driver_node',
        parameters=[LaunchConfiguration('vesc_config')],
    ))
    ld.add_action(Node(
        package='ackermann_mux',
        executable='ackermann_mux',
        name='ackermann_mux',
        parameters=[LaunchConfiguration('mux_config')],
        remappings=[('ackermann_cmd_out', 'ackermann_drive')],
    ))

    return ld
