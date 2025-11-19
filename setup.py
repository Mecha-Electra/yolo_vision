import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'yolo_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Linha para incluir os arquivos de launch da pasta 'launch'
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        # Linha para incluir os modelos da pasta 'resource'
        (os.path.join('share', package_name, 'resource'), glob(os.path.join('resource', '*.pt'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mechaelectra1',
    maintainer_email='your_email@example.com',
    description='A ROS 2 package for YOLO object detection.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector_node = yolo_detector.detector_node:main',
            'crowd_action_server = yolo_detector.crowd_detector_yolo:main', # Renomeado para clareza
            'mission_client = yolo_detector.mission_client:main',
            'person_detector = yolo_detector.person_detector:main',
        ],
    },
)
