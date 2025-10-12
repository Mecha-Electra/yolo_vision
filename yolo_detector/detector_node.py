import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory
import message_filters
from geometry_msgs.msg import PoseWithCovariance, Pose
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import time
from collections import defaultdict

# --- NOVOS IMPORTS ---
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

class ObjectTriggerNode(Node):
    def __init__(self):
        super().__init__('object_detector_node')

        from ultralytics import YOLO

        self.declare_parameter('model', 'yolov8n.pt')
        model_name = self.get_parameter('model').get_parameter_value().string_value
        package_share_directory = get_package_share_directory('yolo_detector')
        model_path = os.path.join(package_share_directory, 'resource', model_name)
        self.get_logger().info(f"Carregando o modelo YOLO de: {model_path}")
        self.model = YOLO(model_path)

        self.bridge = CvBridge()
        self.camera_intrinsics = None
        self.latest_messages = None

        # --- PUBLISHER ATUALIZADO ---
        # Agora publica uma Detection3DArray no tópico 'objects/detections'
        self.objects_publisher = self.create_publisher(Detection3DArray, 'objects/detections', 10)
        self.image_publisher = self.create_publisher(Image, 'objects/annotated_image', 10)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.color_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_raw', qos_profile=qos_profile)
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw', qos_profile=qos_profile)
        self.info_sub = message_filters.Subscriber(self, CameraInfo, '/camera/camera/color/camera_info', qos_profile=qos_profile)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.info_sub], 10, 0.1)
        self.ts.registerCallback(self.image_cache_callback)

        self.trigger_sub = self.create_subscription(
            Empty, 'object_detector/trigger', self.trigger_callback, 10)

        self.is_detecting = False
        self.detection_start_time = 0
        self.frame_count = 0
        self.collected_detections = defaultdict(list)

        self.get_logger().info('Nó Detector de Objetos Robusto iniciado. Aguardando gatilho em /object_detector/trigger')

    def image_cache_callback(self, color_msg, depth_msg, info_msg):
        self.latest_messages = (color_msg, depth_msg, info_msg)
        if self.is_detecting:
            self.process_detection_frame()

    def trigger_callback(self, msg):
        if self.is_detecting:
            self.get_logger().warn("Ciclo de detecção já em andamento. Gatilho ignorado.")
            return
        self.get_logger().info('Gatilho recebido! Iniciando ciclo de detecção de 5 frames...')
        self.is_detecting = True
        self.detection_start_time = time.time()
        self.frame_count = 0
        self.collected_detections.clear()

    def process_detection_frame(self):
        elapsed_time = time.time() - self.detection_start_time
        if elapsed_time > 5.0 or self.frame_count >= 5:
            self.finalize_detection()
            return
        
        if self.latest_messages is None:
            return

        self.frame_count += 1
        self.get_logger().info(f'Processando frame {self.frame_count}/5...')

        color_msg, depth_msg, info_msg = self.latest_messages
        
        if self.camera_intrinsics is None:
            self.camera_intrinsics = np.array(info_msg.k).reshape(3, 3)

        cv_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        
        results = self.model(cv_image, verbose=False)

        for r in results:
            for box in r.boxes:
                confidence = float(box.conf[0])
                if confidence > 0.7:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id]

                    box_radius = 5
                    y_start, y_end = max(0, cy - box_radius), min(depth_image.shape[0], cy + box_radius)
                    x_start, x_end = max(0, cx - box_radius), min(depth_image.shape[1], cx + box_radius)
                    depth_roi = depth_image[y_start:y_end, x_start:x_end]
                    non_zero_depths = depth_roi[depth_roi > 0]

                    if non_zero_depths.size > 0:
                        depth_mm = np.median(non_zero_depths)
                        depth_m = depth_mm / 1000.0
                        fx, fy = self.camera_intrinsics[0, 0], self.camera_intrinsics[1, 1]
                        cx_cam, cy_cam = self.camera_intrinsics[0, 2], self.camera_intrinsics[1, 2]
                        x_cam = (cx - cx_cam) * depth_m / fx
                        y_cam = (cy - cy_cam) * depth_m / fy
                        self.collected_detections[class_name].append((x_cam, y_cam, depth_m))

    # --- FUNÇÃO FINALIZE_DETECTION COMPLETAMENTE ATUALIZADA ---
    def finalize_detection(self):
        self.get_logger().info("Ciclo de detecção concluído. Calculando médias e publicando...")
        self.is_detecting = False

        # Cria a mensagem principal que conterá todas as detecções
        detection_array_msg = Detection3DArray()
        detection_array_msg.header.stamp = self.get_clock().now().to_msg()
        detection_array_msg.header.frame_id = 'camera_color_optical_frame'

        for class_name, detections in self.collected_detections.items():
            if len(detections) >= 2:
                avg_coords = np.mean(detections, axis=0)
                
                self.get_logger().info(f"Objeto confiável '{class_name}' detectado. Posição média: [X: {avg_coords[0]:.2f}, Y: {avg_coords[1]:.2f}, Z: {avg_coords[2]:.2f}]")

                # Cria uma detecção individual para este objeto
                detection = Detection3D()
                detection.header = detection_array_msg.header

                # Cria uma hipótese para o objeto (nome + pose)
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = class_name
                hypothesis.hypothesis.score = 1.0 # Confiança de 100% pois já passou pelo nosso filtro
                
                # Preenche a pose com as coordenadas médias
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = avg_coords
                hypothesis.pose.pose = pose
                
                # Adiciona a hipótese aos resultados da detecção
                detection.results.append(hypothesis)

                # Adiciona esta detecção completa à lista de detecções
                detection_array_msg.detections.append(detection)

        self.objects_publisher.publish(detection_array_msg)
        self.get_logger().info(f"Publicada lista final com {len(detection_array_msg.detections)} objetos confiáveis em /objects/detections")

        if self.latest_messages:
            color_msg, _, _ = self.latest_messages
            cv_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            ros_image_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.image_publisher.publish(ros_image_msg)

        self.get_logger().info("----------------------------------------------------")
        self.get_logger().info("Processo finalizado. Aguardando novo gatilho...")


def main(args=None):
    rclpy.init(args=args)
    node = ObjectTriggerNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()