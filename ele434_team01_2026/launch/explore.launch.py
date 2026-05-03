from launch import LaunchDescription
from launch_ros.actions import Node

from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("tuos_task_sims"),
                    "launch",
                    "obstacle_avoidance.launch.py"
                ])
            )
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("tuos_tb3_tools"),
                    "launch",
                    "slam.launch.py"
                ])),
                launch_arguments={'environment': 'sim'}.items()
        ),
        Node(
            package = 'ele434_team01_2026',
            executable = 'Vacuum.py',
            name = 'my_navigation'
        )
    ])