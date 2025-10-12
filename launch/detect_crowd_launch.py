from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='yolo_detector',
            executable='detector_node',
            name='crowd_detector',
            output='screen',
            parameters=[
                {'model': 'yolov8n.pt'} # Passa o parâmetro para usar o modelo COCO
            ]
        )
    ])
