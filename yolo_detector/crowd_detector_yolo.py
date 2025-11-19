import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter  # Import necessário para parâmetros
from std_msgs.msg import Empty
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
from ultralytics import YOLO
import message_filters
from datetime import datetime
from geometry_msgs.msg import PointStamped, Pose, PoseArray, TransformStamped
import tf2_ros
from tf_transformations import quaternion_from_euler
import os
import cv2

class CrowdTriggerNode(Node):
    def __init__(self):
        super().__init__('crowd_trigger_node')
        self.model = YOLO('yolov8n.pt')
        self.bridge = CvBridge()
        self.camera_intrinsics = None
        self.latest_messages = None

        self.distance_threshold = 3.0

        # Broadcaster TF para publicar as transformadas
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

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

        self.log_file = "yolo_crowd_log.txt"
        self.log_image_dir = "crowd_detections" # Novo diretório para as imagens
        
        # Cria o diretório de log de imagens se ele não existir
        os.makedirs(self.log_image_dir, exist_ok=True)
        self.get_logger().info(f"Imagens de detecção serão salvas em: {self.log_image_dir}")

        with open(self.log_file, "w") as f:
            f.write("Timestamp, Crowd Size, Coordinates of All People (X, Y, Z) meters, Image Filename\n")

    def image_cache_callback(self, color_msg, depth_msg, info_msg):
        self.latest_messages = (color_msg, depth_msg, info_msg)

    def log_crowd_data(self, crowd_size, all_coords, image_filename=None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        coords_str_list = [f"[{c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}]" for c in all_coords]
        
        # Atualização para incluir o nome do arquivo de imagem
        log_entry = f"{timestamp}, {crowd_size}, \"{'; '.join(coords_str_list)}\", {image_filename if image_filename else 'N/A'}\n"
        with open(self.log_file, "a") as f:
            f.write(log_entry)

    def log_bounding_box_image(self, cv_image, detected_people_filtered):
        """Desenha as Bounding Boxes SOMENTE para as pessoas filtradas (dentro do limite) e salva o frame."""
        # Não salva se não houver detecções filtradas
        if not detected_people_filtered:
             return None
             
        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S_%f")
        filename = f"crowd_detection_{timestamp_str}.jpg"
        filepath = os.path.join(self.log_image_dir, filename)

        image_to_save = cv_image.copy()
        color = (0, 255, 0) # BGR: Verde fixo para a multidão filtrada

        for person in detected_people_filtered:
            x1, y1, x2, y2 = person['box']
            coords_3d = person['coords_3d']
            
            # Desenha a caixa delimitadora (Verde)
            cv2.rectangle(image_to_save, (x1, y1), (x2, y2), color, 2)
            
            # Adiciona a etiqueta com a distância (Z)
            text = f"Z: {coords_3d[2]:.2f}m"
            cv2.putText(image_to_save, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Salva a imagem no diretório
        try:
            cv2.imwrite(filepath, image_to_save)
            self.get_logger().info(f"Imagem de detecção salva em: {filepath}")
            return filename
        except Exception as e:
            self.get_logger().error(f"Erro ao salvar a imagem: {e}")
            return None
    # ------------------------------------

    def publish_tf_transforms(self, all_coords, timestamp):
        # A orientação é fixada como identidade (sem rotação)
        # O frame PAI (parent) é o frame da câmera
        parent_frame_id = 'camera_color_optical_frame'
        
        # Usamos um quaternion de identidade (0, 0, 0, 1) para rotação nula
        q = quaternion_from_euler(0, 0, 0)
        
        for i, coords in enumerate(all_coords):
            # O frame FILHO (child) será um identificador único para cada pessoa
            child_frame_id = f'person_{i}'
            
            t = TransformStamped()
            t.header.stamp = timestamp
            t.header.frame_id = parent_frame_id
            t.child_frame_id = child_frame_id
            
            # Posição (translação)
            t.transform.translation.x = coords[0]
            t.transform.translation.y = coords[1]
            t.transform.translation.z = coords[2]
            
            # Orientação (rotação)
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]
            
            self.tf_broadcaster.sendTransform(t)
            
        if all_coords:
            self.get_logger().info(f"Publicando {len(all_coords)} transformadas TF (person_0 a person_{len(all_coords)-1})")
    # ------------------------------------

    def trigger_callback(self, msg):
        self.get_logger().info('Gatilho recebido! Executando detecção one-shot...')

        if self.latest_messages is None:
            self.get_logger().warn("Nenhum frame da câmera recebido ainda. Não é possível processar.")
            return

        color_msg, depth_msg, info_msg = self.latest_messages

        current_time = self.get_clock().now().to_msg() 

        if self.camera_intrinsics is None:
            self.camera_intrinsics = np.array(info_msg.k).reshape(3, 3)

        cv_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        results = self.model(cv_image)

        detected_people_raw = []
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
                            detected_people_raw.append({'box': (x1, y1, x2, y2), 'coords_3d': (x_cam, y_cam, depth_m)})

        # Filtra as pessoas detectadas para incluir apenas aquelas dentro do limite de distância (eixos Z)
        detected_people = [
            p for p in detected_people_raw 
            if p['coords_3d'][2] <= self.distance_threshold
        ]
        detected_people.sort(key=lambda p: p['box'][2], reverse=True)

        crowd_size_raw = len(detected_people_raw)
        crowd_size = len(detected_people)
        
        self.get_logger().info(f"Pessoas detectadas (antes do filtro): {crowd_size_raw}. Pessoas válidas (até {self.distance_threshold:.2f}m): {crowd_size}")

        image_filename = self.log_bounding_box_image(cv_image, detected_people)

        all_coords = [p['coords_3d'] for p in detected_people]
        self.log_crowd_data(crowd_size, all_coords, image_filename)

        self.publish_tf_transforms(all_coords, current_time)

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
            # Encontra a pessoa mais à direita ENTRE AS PESSOAS FILTRADAS
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