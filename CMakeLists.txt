cmake_minimum_required(VERSION 3.8)
project(yolo_detector)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

ament_package()
