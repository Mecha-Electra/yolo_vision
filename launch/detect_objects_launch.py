from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='yolo_detector',
            executable='detector_node',
            name='object_detector',
            output='screen',
            parameters=[
                {'model': 'best.pt'} # Passa o parâmetro para usar o seu futuro modelo customizado
            ]
        )
    ])
