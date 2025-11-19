import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():

    pkg_realsense = get_package_share_directory('realsense2_camera')

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_realsense, 'launch', 'rs_launch.py')
        ]),
        launch_arguments={
            'align_depth.enable': 'true'
        }.items(),
    )

    objects_detector = Node(
        package='yolo_detector',
        executable='detector_node',
        name='object_detector',
        output='screen',
        parameters=[
            {'model': 'best.pt'}
        ]
    )

    # Retorna ambos os lançamentos
    return LaunchDescription([
        realsense_launch,
        objects_detector
    ])
