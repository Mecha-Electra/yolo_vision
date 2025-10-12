import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
from ultralytics import YOLO
import message_filters
from datetime import datetime
from geometry_msgs.msg import PointStamped, Pose, PoseArray

class CrowdTriggerNode(Node):
    def __init__(self):
        super().__init__('crowd_trigger_node')
        self.model = YOLO('yolov8n.pt')
        self.bridge = CvBridge()
        self.camera_intrinsics = None
        self.latest_messages = None

        # Publishers para os resultados
        self.target_publisher = self.create_publisher(PointStamped, 'crowd/target_person', 10)
        self.all_people_publisher = self.create_publisher(PoseArray, 'crowd/all_people_poses', 10)

        # Subscribers para os dados da câmera (sempre guardando o último frame)
        self.color_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')
        self.info_sub = message_filters.Subscriber(self, CameraInfo, '/camera/camera/color/camera_info')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.info_sub], 10, 0.1)
        self.ts.registerCallback(self.image_cache_callback)

        # Subscriber para o gatilho da missão
        self.trigger_sub = self.create_subscription(
            Empty, 'crowd_detector/trigger', self.trigger_callback, 10)

        self.get_logger().info('Nó Detector "On-Demand" iniciado. Aguardando gatilho no tópico /crowd_detector/trigger')
        self.log_file = "yolo_crowd_log.txt"
        with open(self.log_file, "w") as f:
            f.write("Timestamp, Crowd Size, Coordinates of All People (X, Y, Z) meters\n")

    def image_cache_callback(self, color_msg, depth_msg, info_msg):
        self.latest_messages = (color_msg, depth_msg, info_msg)

    def log_crowd_data(self, crowd_size, all_coords):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        coords_str_list = [f"[{c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}]" for c in all_coords]
        log_entry = f"{timestamp}, {crowd_size}, \"{'; '.join(coords_str_list)}\"\n"
        with open(self.log_file, "a") as f:
            f.write(log_entry)

    def trigger_callback(self, msg):
        self.get_logger().info('Gatilho recebido! Executando detecção one-shot...')

        if self.latest_messages is None:
            self.get_logger().warn("Nenhum frame da câmera recebido ainda. Não é possível processar.")
            return

        color_msg, depth_msg, info_msg = self.latest_messages

        if self.camera_intrinsics is None:
            self.camera_intrinsics = np.array(info_msg.k).reshape(3, 3)

        cv_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        results = self.model(cv_image)

        detected_people = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]
                if class_name == 'person':
                    confidence = float(box.conf[0])
                    if confidence > 0.6:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2

                        # --- LÓGICA DE PROFUNDIDADE ROBUSTA ---
                        box_radius = 5
                        y_start, y_end = max(0, cy - box_radius), min(depth_image.shape[0], cy + box_radius)
                        x_start, x_end = max(0, cx - box_radius), min(depth_image.shape[1], cx + box_radius)
                        depth_roi = depth_image[y_start:y_end, x_start:x_end]
                        non_zero_depths = depth_roi[depth_roi > 0]

                        if non_zero_depths.size > 0:
                            depth_mm = np.median(non_zero_depths)
                        else:
                            depth_mm = 0
                        # --- FIM DA LÓGICA ROBUSTA ---

                        if depth_mm > 0:
                            depth_m = depth_mm / 1000.0
                            fx, fy = self.camera_intrinsics[0, 0], self.camera_intrinsics[1, 1]
                            cx_cam, cy_cam = self.camera_intrinsics[0, 2], self.camera_intrinsics[1, 2]
                            x_cam = (cx - cx_cam) * depth_m / fx
                            y_cam = (cy - cy_cam) * depth_m / fy
                            detected_people.append({'box': (x1, y1, x2, y2), 'coords_3d': (x_cam, y_cam, depth_m)})

        crowd_size = len(detected_people)
        self.get_logger().info(f"Total de pessoas com 3D válido: {crowd_size}")

        all_coords = [p['coords_3d'] for p in detected_people]
        self.log_crowd_data(crowd_size, all_coords)

        all_poses_msg = PoseArray()
        all_poses_msg.header.stamp = self.get_clock().now().to_msg()
        all_poses_msg.header.frame_id = 'camera_color_optical_frame'
        for coords in all_coords:
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = coords
            all_poses_msg.poses.append(pose)
        self.all_people_publisher.publish(all_poses_msg)
        self.get_logger().info(f"Publicando lista com {crowd_size} pessoas em /crowd/all_people_poses")

        if crowd_size > 0:
            rightmost_person = max(detected_people, key=lambda p: p['box'][2])
            target_coords = rightmost_person['coords_3d']

            point_msg = PointStamped()
            point_msg.header.stamp = self.get_clock().now().to_msg()
            point_msg.header.frame_id = 'camera_color_optical_frame'
            point_msg.point.x, point_msg.point.y, point_msg.point.z = target_coords
            self.target_publisher.publish(point_msg)

            self.get_logger().info(f"Publicando ALVO em /crowd/target_person: X={target_coords[0]:.2f}, Y={target_coords[1]:.2f}, Z={target_coords[2]:.2f}")

        self.get_logger().info('Detecção one-shot concluída.')

def main(args=None):
    rclpy.init(args=args)
    node = CrowdTriggerNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
