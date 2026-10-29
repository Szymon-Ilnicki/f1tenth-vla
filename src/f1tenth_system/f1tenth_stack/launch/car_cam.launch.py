from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
import os

#This launch scripts extends on car_min.launch by adding camera module

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

    task_latch_node = Node(
        package='f1tenth_stack',          # package that contains task_latch.py
        executable='task_latch.py',           # or 'task_latch' if you installed it as an entry point
        name='task_latch',
        output='screen'
    )

    return LaunchDescription([car, cam, task_latch_node])
