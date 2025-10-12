from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='yolo_detector',
            executable='crowd_yolo',  # <-- A MUDANÇA CRÍTICA ESTÁ AQUI
            name='crowd_detector_oneshot',
            output='screen'
        )
    ])
