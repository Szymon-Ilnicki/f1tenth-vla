from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def launch_setup(context):
    pkg = FindPackageShare('f1tenth_stack')
    config_choice = context.launch_configurations.get('map_config', 'normal')
    
    if config_choice == 'tiny':
        config_file = 'tiny_localization.yaml'
    else:
        config_file = 'slam_localization.yaml'
    
    return [
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='both',
            parameters=[
                PathJoinSubstitution([pkg, 'config', config_file]),
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ],
            remappings=[('/scan', '/scan')]
        )
    ]

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('map_config', default_value='normal'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        OpaqueFunction(function=launch_setup),
    ])
